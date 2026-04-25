#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "wst_kernel.cuh"
#include "jtfs_kernel.cuh"
#include "memory_staging.cuh"

namespace py = pybind11;

// Entry point fingerprint function
py::array_t<float> fingerprint(py::array_t<float> signal, const WSTConfig& cfg) {
    // Request a buffer descriptor from Python
    py::buffer_info buf = signal.request();
    
    if (buf.ndim != 1) {
        throw std::runtime_error("Input signal must be 1-D");
    }
    
    int signal_len = buf.shape[0];
    if (signal_len != cfg.signal_len) {
        throw std::runtime_error("Signal length does not match config");
    }
    
    // Zero-copy pointer access
    float* ptr = static_cast<float*>(buf.ptr);
    
    // Instantiate staging (assuming batch_size=1 for simple entrypoint, or from cfg)
    MemoryStaging staging(cfg.signal_len, cfg.batch_size, cfg.q);
    staging.load_input(ptr);
    staging.transfer_to_device_async();
    
    // Output calculation
    size_t out_elements = cfg.signal_len * cfg.batch_size; // simplified representation
    
    if (cfg.jtfs) {
        JTFSConfig jtfs_cfg { 8, 16 }; // Defaults for demonstration
        // For Hopper
        JTFSEngine<HopperTag, 8, 16, 8> engine(cfg, jtfs_cfg);
        // We mock the launch here since we don't have all intermediate buffers in the simple wrapper
    } else {
        // For Hopper
        WSTEngine<HopperTag, 8, 16> engine(cfg);
        engine.set_stream(staging.get_compute_stream());
        engine.forward_pass(staging.get_device_input(), staging.get_device_output(), nullptr, staging.get_compute_stream());
    }
    
    staging.transfer_to_host_async();
    
    // Create numpy array to return, copying from pinned memory
    auto result = py::array_t<float>(out_elements);
    py::buffer_info res_buf = result.request();
    float* res_ptr = static_cast<float*>(res_buf.ptr);
    memcpy(res_ptr, staging.get_host_output(), out_elements * sizeof(float));
    
    return result;
}

PYBIND11_MODULE(_core, m) {
    m.doc() = "omni-wst-core C++/CUDA Mathematical Primitives";
    
    py::class_<WSTConfig>(m, "WSTConfig")
        .def(py::init<int, int, int, int, int, bool, float>(),
             py::arg("signal_len"),
             py::arg("batch_size"),
             py::arg("j"),
             py::arg("q"),
             py::arg("depth"),
             py::arg("jtfs"),
             py::arg("l1_norm_psi"))
        .def_readwrite("signal_len", &WSTConfig::signal_len)
        .def_readwrite("batch_size", &WSTConfig::batch_size)
        .def_readwrite("j", &WSTConfig::j)
        .def_readwrite("q", &WSTConfig::q)
        .def_readwrite("depth", &WSTConfig::depth)
        .def_readwrite("jtfs", &WSTConfig::jtfs)
        .def_readwrite("l1_norm_psi", &WSTConfig::l1_norm_psi);
        
    py::class_<JTFSConfig>(m, "JTFSConfig")
        .def(py::init<int, int>(),
             py::arg("J_fr"),
             py::arg("Q_fr"))
        .def_readwrite("J_fr", &JTFSConfig::J_fr)
        .def_readwrite("Q_fr", &JTFSConfig::Q_fr);
        
    m.def("fingerprint", &fingerprint, "Compute WST/JTFS fingerprint");
}
