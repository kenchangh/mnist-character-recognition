/*
 * tt_um_kenchangh_mnist — self-contained MNIST digit recognition.
 *
 * 8x8 binarized image is shifted in serially (64 bits, pixel 0 first),
 * then classified by a ternary-weight network whose weights are Verilog
 * constants (constant-folded at synthesis — no on-chip weight storage).
 *
 * Variants (parameters MLP, HIDDEN):
 *   MLP = 0             : linear classifier 64 -> 10
 *   MLP = 1, HIDDEN = H : MLP 64 -> H -> 10 (H = 16 or 32),
 *                         binary {0,1} hidden activations
 *
 * Compute is bit-serial: one pixel (or hidden bit) per cycle feeds up to 16
 * parallel add/sub/skip accumulator lanes, then a sequential argmax.
 * HIDDEN = 32 runs layer 1 as two 16-lane passes over the (restored) input
 * shift register, halving the accumulator flops at zero accuracy cost.
 * Latency after the 64th input bit:
 *   linear 74, mlp16 91, mlp32 172 cycles  (~17 us total at 10 MHz).
 *
 * Interface (see docs/info.md for the timing diagram):
 *   ui_in[0]  DATA      serial pixel bit
 *   ui_in[1]  SHIFT_EN  sample DATA on rising clk edge when high
 *   ui_in[2]  START     strobe: begin a new image (clears bit counter)
 *   uo_out[6:0]  7-segment a..g of last classified digit
 *   uo_out[7]    DONE (also lights the 7-seg decimal point)
 *   uio_out[3:0] predicted digit (binary)
 *   uio_out[4]   DONE
 *   uio_out[5]   BUSY (computing)
 *
 * Copyright (c) 2026 kenchangh
 * SPDX-License-Identifier: Apache-2.0
 */

`default_nettype none

module tt_um_kenchangh_mnist #(
    parameter MLP    = 1,  // 0: linear 64->10, 1: MLP 64->HIDDEN->10
    parameter HIDDEN = 32  // hidden units (16 or 32), used when MLP = 1
) (
    input  wire [7:0] ui_in,    // Dedicated inputs
    output wire [7:0] uo_out,   // Dedicated outputs
    input  wire [7:0] uio_in,   // IOs: Input path
    output wire [7:0] uio_out,  // IOs: Output path
    output wire [7:0] uio_oe,   // IOs: Enable path (active high: 0=input, 1=output)
    input  wire       ena,      // always 1 when the design is powered, so you can ignore it
    input  wire       clk,      // clock
    input  wire       rst_n     // reset_n - low to reset
);

  // Layer 1 runs in PASSES passes over LANES parallel accumulator lanes.
  localparam LANES   = MLP ? (HIDDEN > 16 ? 16 : HIDDEN) : 10;
  localparam PASSES  = (MLP != 0 && HIDDEN > 16) ? 2 : 1;
  localparam [5:0] L2_LAST = HIDDEN - 1;

  localparam [2:0] S_IDLE   = 3'd0,
                   S_LOAD   = 3'd1,  // shifting in 64 pixel bits
                   S_L1     = 3'd2,  // 64-cycle bit-serial layer 1
                   S_BIN    = 3'd3,  // binarize hidden, preload layer-2 bias
                   S_L2     = 3'd4,  // 16-cycle bit-serial layer 2 (MLP only)
                   S_ARGMAX = 3'd5,  // 10-cycle sequential argmax
                   S_DONE   = 3'd6;

  wire data_in  = ui_in[0];
  wire shift_en = ui_in[1];
  wire start    = ui_in[2];

  reg  [2:0]        state;
  reg  [5:0]        cnt;
  reg  [63:0]       inp;         // input shift register, bit i = pixel i
  reg  signed [7:0] acc [0:15];  // shared layer-1 / layer-2 accumulator lanes
  reg  [31:0]       h;           // binarized hidden activations (MLP)
  reg               pass;        // layer-1 pass index (mlp32: 0 then 1)
  reg  signed [7:0] best_val;
  reg  [3:0]        best_idx;
  reg  [3:0]        digit;
  reg               done_r;
  reg               have_result; // 7-seg stays blank until first result

  // Weight ROMs (generated constants; synthesis folds them into logic)
  wire [31:0]  l1_pos, l1_neg;
  wire [255:0] b1_flat;
  wire [9:0]   l2_pos, l2_neg;
  wire [79:0]  b2_flat;

  generate
    if (MLP != 0 && HIDDEN == 32) begin : g_w
      mnist_weights_mlp32 u_weights (
          .idx(cnt), .hidx(cnt[4:0]),
          .l1_pos(l1_pos), .l1_neg(l1_neg), .b1_flat(b1_flat),
          .l2_pos(l2_pos), .l2_neg(l2_neg), .b2_flat(b2_flat));
    end else if (MLP != 0) begin : g_w
      mnist_weights_mlp16 u_weights (
          .idx(cnt), .hidx(cnt[4:0]),
          .l1_pos(l1_pos), .l1_neg(l1_neg), .b1_flat(b1_flat),
          .l2_pos(l2_pos), .l2_neg(l2_neg), .b2_flat(b2_flat));
    end else begin : g_w
      mnist_weights_linear u_weights (
          .idx(cnt), .hidx(cnt[4:0]),
          .l1_pos(l1_pos), .l1_neg(l1_neg), .b1_flat(b1_flat),
          .l2_pos(l2_pos), .l2_neg(l2_neg), .b2_flat(b2_flat));
    end
  endgenerate

  wire pix = inp[0];  // pixel `cnt` sits at bit 0 during S_L1 (rotation)

  // weight columns for the lanes of the current pass
  wire [15:0] w1p_sel = pass ? l1_pos[31:16] : l1_pos[15:0];
  wire [15:0] w1n_sel = pass ? l1_neg[31:16] : l1_neg[15:0];

  integer i;

  always @(posedge clk) begin
    if (!rst_n) begin
      state       <= S_IDLE;
      cnt         <= 6'd0;
      inp         <= 64'd0;
      h           <= 32'd0;
      best_val    <= 8'sd0;
      best_idx    <= 4'd0;
      digit       <= 4'd0;
      done_r      <= 1'b0;
      have_result <= 1'b0;
      pass        <= 1'b0;
      for (i = 0; i < 16; i = i + 1) acc[i] <= 8'sd0;
    end else if (start) begin
      // START restarts cleanly from any state, including mid-load
      state  <= S_LOAD;
      cnt    <= 6'd0;
      pass   <= 1'b0;
      done_r <= 1'b0;
    end else begin
      case (state)
        S_LOAD: begin
          if (shift_en) begin
            inp <= {data_in, inp[63:1]};
            cnt <= cnt + 6'd1;
            if (cnt == 6'd63) begin
              // 64th bit received: preload layer-1 biases, start compute
              state <= S_L1;
              cnt   <= 6'd0;
              pass  <= 1'b0;
              for (i = 0; i < LANES; i = i + 1)
                acc[i] <= $signed(b1_flat[i*8 +: 8]);
            end
          end
        end

        S_L1: begin
          inp <= {inp[0], inp[63:1]};  // rotate: restored after 64 cycles
          for (i = 0; i < LANES; i = i + 1) begin
            if (pix && w1p_sel[i])      acc[i] <= acc[i] + 8'sd1;
            else if (pix && w1n_sel[i]) acc[i] <= acc[i] - 8'sd1;
          end
          cnt <= cnt + 6'd1;
          if (cnt == 6'd63) begin
            cnt <= 6'd0;
            if (MLP != 0) begin
              state <= S_BIN;
            end else begin
              state    <= S_ARGMAX;
              best_val <= -8'sd128;
              best_idx <= 4'd0;
            end
          end
        end

        S_BIN: begin  // MLP only: h = (acc >= 0), then next pass or layer 2
          for (i = 0; i < LANES; i = i + 1) begin
            if (pass == 1'b0) h[i] <= ~acc[i][7];
            else              h[16 + i] <= ~acc[i][7];
          end
          cnt <= 6'd0;
          if (PASSES == 2 && pass == 1'b0) begin
            // second layer-1 pass: units 16..31, biases from upper half
            pass  <= 1'b1;
            state <= S_L1;
            for (i = 0; i < LANES; i = i + 1)
              acc[i] <= $signed(b1_flat[(i + 16)*8 +: 8]);
          end else begin
            state <= S_L2;
            for (i = 0; i < 10; i = i + 1)
              acc[i] <= $signed(b2_flat[i*8 +: 8]);
          end
        end

        S_L2: begin  // MLP only: one hidden bit per cycle
          for (i = 0; i < 10; i = i + 1) begin
            if (h[cnt[4:0]] && l2_pos[i])      acc[i] <= acc[i] + 8'sd1;
            else if (h[cnt[4:0]] && l2_neg[i]) acc[i] <= acc[i] - 8'sd1;
          end
          cnt <= cnt + 6'd1;
          if (cnt == L2_LAST) begin
            cnt      <= 6'd0;
            state    <= S_ARGMAX;
            best_val <= -8'sd128;
            best_idx <= 4'd0;
          end
        end

        S_ARGMAX: begin  // first (lowest-index) maximum wins ties
          if (acc[cnt[3:0]] > best_val) begin
            best_val <= acc[cnt[3:0]];
            best_idx <= cnt[3:0];
          end
          cnt <= cnt + 6'd1;
          if (cnt == 6'd9) begin
            digit       <= (acc[9] > best_val) ? 4'd9 : best_idx;
            done_r      <= 1'b1;
            have_result <= 1'b1;
            state       <= S_DONE;
          end
        end

        S_DONE: ;  // hold result until next START

        default: state <= S_IDLE;
      endcase
    end
  end

  // 7-segment decode (TT demo board: uo[0]=a ... uo[6]=g, uo[7]=dp)
  reg [6:0] seg;
  always @(*) begin
    case (digit)
      4'd0: seg = 7'h3F;
      4'd1: seg = 7'h06;
      4'd2: seg = 7'h5B;
      4'd3: seg = 7'h4F;
      4'd4: seg = 7'h66;
      4'd5: seg = 7'h6D;
      4'd6: seg = 7'h7D;
      4'd7: seg = 7'h07;
      4'd8: seg = 7'h7F;
      4'd9: seg = 7'h6F;
      default: seg = 7'h00;
    endcase
  end

  wire busy = (state == S_L1) || (state == S_BIN) ||
              (state == S_L2) || (state == S_ARGMAX);

  assign uo_out  = {done_r, have_result ? seg : 7'h00};
  assign uio_out = {2'b00, busy, done_r, digit};
  assign uio_oe  = 8'h3F;

  // List all unused inputs to prevent warnings
  wire _unused = &{ena, ui_in[7:3], uio_in, 1'b0};

endmodule
