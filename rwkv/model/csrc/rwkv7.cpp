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
    }
}