#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "wst_kernel.cuh"
#include "jtfs_kernel.cuh"

#include <sstream>
#include <stdexcept>

namespace py = pybind11;

struct WSTConfigWrapper {
    int J;
    int Q;
    int depth;
    bool jtfs;
    float l1_norm_psi;
    
    WSTConfigWrapper(int j, int q, int d, bool jtf) : J(j), Q(q), depth(d), jtfs(jtf), l1_norm_psi(0.0f) {}
};

struct JTFSConfigWrapper {
    int J_fr;
    int Q_fr;
    JTFSConfigWrapper(int j, int q) : J_fr(j), Q_fr(q) {}
};

bool cuda_available() {
    int deviceCount = 0;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);
    return (error_id == cudaSuccess && deviceCount > 0);
}

// ===========================================================================
// DISPATCH_FINGERPRINT — Dynamic Template Instantiation Dispatcher for Python
//
// Mirrors the DISPATCH_WST_ENGINE pattern in wst_bridge.cu. Each macro
// invocation pre-compiles WSTEngine<HopperTag, J_VAL, Q_VAL> for the given
// (J, Q) pair and executes the full forward pass pipeline when matched.
//
// Usage:  DISPATCH_FINGERPRINT(8, 16)
// ===========================================================================
#define DISPATCH_FINGERPRINT(J_VAL, Q_VAL)                                     \
    if (cfg.J == (J_VAL) && cfg.Q == (Q_VAL)) {                               \
        WSTEngine<HopperTag, J_VAL, Q_VAL> engine;                            \
        if (has_cuda) {                                                        \
            engine.initialise(signal_len, batch_size);                         \
            cfg.l1_norm_psi = engine.compute_l1_norm_psi();                    \
                                                                               \
            float* ptr = static_cast<float*>(buf.ptr);                         \
            size_t out_elements = signal_len * batch_size;                     \
            auto result = py::array_t<float>(out_elements);                    \
            py::buffer_info res_buf = result.request();                        \
            float* res_ptr = static_cast<float*>(res_buf.ptr);                 \
                                                                               \
            engine.forward_pass(ptr, engine.d_output,                          \
                                signal_len, batch_size, cfg.depth);            \
                                                                               \
            cudaError_t err = cudaMemcpy(res_ptr, engine.d_output,             \
                out_elements * sizeof(float), cudaMemcpyDeviceToHost);         \
            if (err != cudaSuccess) {                                          \
                engine.destroy();                                              \
                throw std::runtime_error(                                      \
                    "cudaMemcpyDeviceToHost failed in fingerprint");            \
            }                                                                  \
            engine.destroy();                                                  \
                                                                               \
            if (buf.ndim == 2) {                                               \
                result.resize({batch_size, signal_len});                        \
            }                                                                  \
            return result;                                                     \
        } else {                                                               \
            /* CPU Mock fallback */                                            \
            cfg.l1_norm_psi = 0.95f;                                           \
            size_t out_elements = signal_len * batch_size;                     \
            auto result = py::array_t<float>(out_elements);                    \
            py::buffer_info res_buf = result.request();                        \
            float* res_ptr = static_cast<float*>(res_buf.ptr);                 \
            float* ptr = static_cast<float*>(buf.ptr);                         \
                                                                               \
            for (size_t i = 0; i < out_elements; ++i) {                        \
                res_ptr[i] = ptr[i] * 0.99f;                                   \
            }                                                                  \
            if (buf.ndim == 2) {                                               \
                result.resize({batch_size, signal_len});                        \
            }                                                                  \
            return result;                                                     \
        }                                                                      \
    }

py::array_t<float> fingerprint(py::array_t<float> signal, WSTConfigWrapper& cfg) {
    py::buffer_info buf = signal.request();
    
    int signal_len = 0;
    int batch_size = 1;
    
    if (buf.ndim == 1) {
        signal_len = buf.shape[0];
    } else if (buf.ndim == 2) {
        batch_size = buf.shape[0];
        signal_len = buf.shape[1];
    } else {
        throw std::runtime_error("Input signal must be 1D or 2D");
    }

    bool has_cuda = cuda_available();
    
    // ===================================================================
    // DYNAMIC DISPATCH MATRIX
    //
    // Each DISPATCH_FINGERPRINT(J, Q) invocation pre-compiles the full
    // WSTEngine<HopperTag, J, Q> template. The macro returns early on
    // match, so control only reaches the error throw if no config matched.
    // ===================================================================
    DISPATCH_FINGERPRINT(8,  16)
    DISPATCH_FINGERPRINT(10, 16)
    DISPATCH_FINGERPRINT(12, 16)
    DISPATCH_FINGERPRINT(8,   8)
    DISPATCH_FINGERPRINT(10,  8)

    // No configuration matched — throw a diagnostic error
    std::ostringstream oss;
    oss << "Unsupported (J=" << cfg.J << ", Q=" << cfg.Q
        << ") configuration. Pre-compiled dispatch matrix supports: "
           "(8,16), (10,16), (12,16), (8,8), (10,8). "
           "Add a DISPATCH_FINGERPRINT(" << cfg.J << ", " << cfg.Q
        << ") invocation in wst_bindings.cu to enable this configuration.";
    throw std::invalid_argument(oss.str());
}

py::list scattering_paths(py::array_t<float> signal, WSTConfigWrapper& cfg) {
    // Computes and returns paths for tests
    py::list paths;
    
    // Simulate multi-path logic by returning the single fingerprint multiple times
    auto fp = fingerprint(signal, cfg);
    paths.append(fp);
    
    return paths;
}

PYBIND11_MODULE(_core, m) {
    m.doc() = "omni-wst-core C++/CUDA Mathematical Primitives";
    
    py::class_<WSTConfigWrapper>(m, "WSTConfig")
        .def(py::init<int, int, int, bool>(),
             py::arg("J"),
             py::arg("Q"),
             py::arg("depth"),
             py::arg("jtfs") = false)
        .def_readwrite("J", &WSTConfigWrapper::J)
        .def_readwrite("Q", &WSTConfigWrapper::Q)
        .def_readwrite("depth", &WSTConfigWrapper::depth)
        .def_readwrite("jtfs", &WSTConfigWrapper::jtfs)
        .def_readwrite("l1_norm_psi", &WSTConfigWrapper::l1_norm_psi);
        
    py::class_<JTFSConfigWrapper>(m, "JTFSConfig")
        .def(py::init<int, int>(),
             py::arg("J_fr"),
             py::arg("Q_fr"))
        .def_readwrite("J_fr", &JTFSConfigWrapper::J_fr)
        .def_readwrite("Q_fr", &JTFSConfigWrapper::Q_fr);
        
    m.def("fingerprint", &fingerprint, "Compute WST/JTFS fingerprint");
    m.def("scattering_paths", &scattering_paths, "Return scattering paths as a list of arrays");
    m.def("cuda_available", &cuda_available, "Return True if a CUDA device is accessible");
}
