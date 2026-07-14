`default_nettype none
`timescale 1ns / 1ps

/* Testbench wrapper: instantiates the MNIST classifier and provides wires
   driven / sampled by the cocotb tests in test.py.

   Variant selection at compile time (make VARIANT=...):
     -DVARIANT_LINEAR  linear 64->10
     -DVARIANT_MLP16   MLP 64->16->10
     (default)         MLP 64->32->10
   Gate-level sim (-DGL_TEST) uses the hardened netlist, which has no
   parameters — it is whatever variant src/project.v defaults to. */
module tb ();

  // Dump the signals to a FST file. You can view it with gtkwave or surfer.
  initial begin
    $dumpfile("tb.fst");
    $dumpvars(0, tb);
    #1;
  end

  // Wire up the inputs and outputs:
  reg clk;
  reg rst_n;
  reg ena;
  reg [7:0] ui_in;
  reg [7:0] uio_in;
  wire [7:0] uo_out;
  wire [7:0] uio_out;
  wire [7:0] uio_oe;

`ifdef GL_TEST
  tt_um_guanhao3797_mnist user_project (
`elsif VARIANT_LINEAR
  tt_um_guanhao3797_mnist #(.MLP(0)) user_project (
`elsif VARIANT_MLP16
  tt_um_guanhao3797_mnist #(.MLP(1), .HIDDEN(16)) user_project (
`else
  tt_um_guanhao3797_mnist #(.MLP(1), .HIDDEN(32)) user_project (
`endif
      .ui_in  (ui_in),    // Dedicated inputs
      .uo_out (uo_out),   // Dedicated outputs
      .uio_in (uio_in),   // IOs: Input path
      .uio_out(uio_out),  // IOs: Output path
      .uio_oe (uio_oe),   // IOs: Enable path (active high: 0=input, 1=output)
      .ena    (ena),      // enable - goes high when design is selected
      .clk    (clk),      // clock
      .rst_n  (rst_n)     // not reset
  );

endmodule
