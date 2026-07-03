#include <Python.h>
#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>
#include <vector>

extern "C" {
  // PyMODINIT_FUNC adds __declspec(dllexport) on Windows so the loader can find
  // PyInit_RWKV_CUDA. Upstream's raw `extern "C" PyObject*` only exports on Linux
  // (default symbol visibility); on Windows the symbol stays hidden and the .pyd
  // import fails (ops still register via TORCH_LIBRARY static-init, but the Python
  // module object would be None). This makes the import succeed cleanly on Windows.
  PyMODINIT_FUNC PyInit_RWKV_CUDA(void)
  {
      static struct PyModuleDef module_def = {
          PyModuleDef_HEAD_INIT,
          "RWKV_CUDA",
          NULL,
          -1,
          NULL,
      };
      return PyModule_Create(&module_def);
  }
}

namespace rwkv {
    TORCH_LIBRARY(rwkv, m) {
        m.def("rwkv7_wkv_forward_float(Tensor r_BTHK, Tensor k_BTHK, Tensor v_BTHK, Tensor w_BTHK, Tensor a_BTHK, Tensor k_deformed_BTHK, Tensor skip_BTH) -> (Tensor, Tensor)");
        m.def("rwkv7_wkv_backward_float(Tensor r_BTHK, Tensor k_BTHK, Tensor v_BTHK, Tensor w_BTHK, Tensor a_BTHK, Tensor k_deformed_BTHK, Tensor skip_BTH, Tensor state_checkpoints_BLHKK, Tensor grad_BTHK) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");
        m.def("rwkv7_wkv_forward_bfloat16(Tensor r_BTHK, Tensor k_BTHK, Tensor v_BTHK, Tensor w_BTHK, Tensor a_BTHK, Tensor k_deformed_BTHK, Tensor skip_BTH) -> (Tensor, Tensor)");
        m.def("rwkv7_wkv_backward_bfloat16(Tensor r_BTHK, Tensor k_BTHK, Tensor v_BTHK, Tensor w_BTHK, Tensor a_BTHK, Tensor k_deformed_BTHK, Tensor skip_BTH, Tensor state_checkpoints_BLHKK, Tensor grad_BTHK) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");
        m.def("rwkv7_wkv_forward_half(Tensor r_BTHK, Tensor k_BTHK, Tensor v_BTHK, Tensor w_BTHK, Tensor a_BTHK, Tensor k_deformed_BTHK, Tensor skip_BTH) -> (Tensor, Tensor)");
        m.def("rwkv7_wkv_backward_half(Tensor r_BTHK, Tensor k_BTHK, Tensor v_BTHK, Tensor w_BTHK, Tensor a_BTHK, Tensor k_deformed_BTHK, Tensor skip_BTH, Tensor state_checkpoints_BLHKK, Tensor grad_BTHK) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");
        // Stateful BPTT variants: forward takes an initial state state0_BHKK [B,H,K,K] fp32 and also
        // returns the final state [B,H,K,K] fp32; backward forces the sequential kernel (the saved
        // checkpoint[0] = state0, so it is correct for a nonzero start). Truncated BPTT: no grad to state0.
        m.def("rwkv7_wkv_forward_stateful_float(Tensor r_BTHK, Tensor k_BTHK, Tensor v_BTHK, Tensor w_BTHK, Tensor a_BTHK, Tensor k_deformed_BTHK, Tensor skip_BTH, Tensor state0_BHKK) -> (Tensor, Tensor, Tensor)");
        m.def("rwkv7_wkv_backward_stateful_float(Tensor r_BTHK, Tensor k_BTHK, Tensor v_BTHK, Tensor w_BTHK, Tensor a_BTHK, Tensor k_deformed_BTHK, Tensor skip_BTH, Tensor state_checkpoints_BLHKK, Tensor grad_BTHK) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");
        m.def("rwkv7_wkv_forward_stateful_bfloat16(Tensor r_BTHK, Tensor k_BTHK, Tensor v_BTHK, Tensor w_BTHK, Tensor a_BTHK, Tensor k_deformed_BTHK, Tensor skip_BTH, Tensor state0_BHKK) -> (Tensor, Tensor, Tensor)");
        m.def("rwkv7_wkv_backward_stateful_bfloat16(Tensor r_BTHK, Tensor k_BTHK, Tensor v_BTHK, Tensor w_BTHK, Tensor a_BTHK, Tensor k_deformed_BTHK, Tensor skip_BTH, Tensor state_checkpoints_BLHKK, Tensor grad_BTHK) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");
        m.def("rwkv7_wkv_forward_stateful_half(Tensor r_BTHK, Tensor k_BTHK, Tensor v_BTHK, Tensor w_BTHK, Tensor a_BTHK, Tensor k_deformed_BTHK, Tensor skip_BTH, Tensor state0_BHKK) -> (Tensor, Tensor, Tensor)");
        m.def("rwkv7_wkv_backward_stateful_half(Tensor r_BTHK, Tensor k_BTHK, Tensor v_BTHK, Tensor w_BTHK, Tensor a_BTHK, Tensor k_deformed_BTHK, Tensor skip_BTH, Tensor state_checkpoints_BLHKK, Tensor grad_BTHK) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");
        // Fused QAT: per-step WKV + full-matrix int-N state quant (STE). forward returns (out, quantized
        // state_checkpoints, per-step scale_BT); backward consumes scale_BT so it can stay per-(b,h). qmax=int8:127/int4:7/int2:1.
        m.def("rwkv7_wkv_qat_forward_float(Tensor r_BTHK, Tensor k_BTHK, Tensor v_BTHK, Tensor w_BTHK, Tensor a_BTHK, Tensor k_deformed_BTHK, Tensor skip_BTH, float qmax) -> (Tensor, Tensor, Tensor)");
        m.def("rwkv7_wkv_qat_backward_float(Tensor r_BTHK, Tensor k_BTHK, Tensor v_BTHK, Tensor w_BTHK, Tensor a_BTHK, Tensor k_deformed_BTHK, Tensor skip_BTH, Tensor state_checkpoints_BLHKK, Tensor scale_BT, Tensor grad_BTHK, float qmax) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");
        // Stage-A validation: rank-1 int-N low-rank truncation of a [B,H,K,K] state (matches deploy compress_wkv_state r==1).
        m.def("rwkv7_lr_trunc_test_float(Tensor state_BHKK, float qmax) -> Tensor");
        // Fused rank-1 int-N low-rank QAT (matches deploy rank-1 deploy). forward -> (out, truncated checkpoints).
        m.def("rwkv7_wkv_qat_lr_forward_float(Tensor r_BTHK, Tensor k_BTHK, Tensor v_BTHK, Tensor w_BTHK, Tensor a_BTHK, Tensor k_deformed_BTHK, Tensor skip_BTH, float qmax) -> (Tensor, Tensor)");
        m.def("rwkv7_wkv_qat_lr_backward_float(Tensor r_BTHK, Tensor k_BTHK, Tensor v_BTHK, Tensor w_BTHK, Tensor a_BTHK, Tensor k_deformed_BTHK, Tensor skip_BTH, Tensor state_checkpoints_BLHKK, Tensor grad_BTHK, float qmax) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");
        // Upload the rank-1 PQ codebook to device globals (m<=0 disables PQ -> qat_lr_rank1 uses int-N). Global state.
        m.def("rwkv7_set_pq_codebook(Tensor cb_flat, int m, int sub, int ncent) -> ()");
    }
}