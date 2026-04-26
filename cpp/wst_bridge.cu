// wst_bridge.cu — CUDA implementation of the cxx FFI entry point.
// Compiled by nvcc as part of the `omni_wst_bridge` shared library target.
// This file is intentionally separate from wst_bindings.cu (the Python layer)
// so the two distribution targets (PyPI wheel and Rust sys crate) never interfere.
//
// TDD Reference: Section 2.1 — Zero-Cost cxx FFI Bridge

#include "wst_bridge.h"
#include "wst_kernel.cuh"
#include "jtfs_kernel.cuh"
#include "memory_staging.cuh"

#include <cuda_runtime.h>
#include <chrono>
#include <stdexcept>

// ---------------------------------------------------------------------------
// Internal helper: map a host-side Plasma mmap ptr to a pinned CUDA buffer.
// The Plasma store allocates shared memory via shm_open / mmap. Before the
// kernel can read it, we register the pages as CUDA host memory so UVA
// can address them from device code without an explicit H2D memcpy.
// ---------------------------------------------------------------------------
static float* register_plasma_buffer(uint64_t plasma_ptr, size_t byte_count) {
    void* host_ptr = reinterpret_cast<void*>(plasma_ptr);
    cudaError_t err = cudaHostRegister(host_ptr, byte_count, cudaHostRegisterPortable);
    if (err != cudaSuccess && err != cudaErrorHostMemoryAlreadyRegistered) {
        throw std::runtime_error(
            std::string("cudaHostRegister failed: ") + cudaGetErrorString(err));
    }
    return static_cast<float*>(host_ptr);
}

static void unregister_plasma_buffer(uint64_t plasma_ptr) {
    void* host_ptr = reinterpret_cast<void*>(plasma_ptr);
    // Best-effort unregister — if it was already registered elsewhere, ignore.
    cudaHostUnregister(host_ptr);
}

// ---------------------------------------------------------------------------
// run_wst_pipeline — Primary FFI entry point called by the Rust orchestrator.
// ---------------------------------------------------------------------------
WSTResult run_wst_pipeline(
    uint64_t input_plasma_ptr,
    int32_t  signal_len,
    int32_t  batch_size,
    int32_t  J,
    int32_t  Q,
    int32_t  depth,
    bool     use_jtfs
) {
    // --- Timing start ---
    auto t_start = std::chrono::high_resolution_clock::now();

    // --- Validate parameters ---
    if (signal_len <= 0 || batch_size <= 0 || J <= 0 || Q <= 0 || depth <= 0) {
        throw std::runtime_error("run_wst_pipeline: invalid configuration parameters");
    }

    const size_t input_bytes  = static_cast<size_t>(signal_len) * batch_size * sizeof(float);
    const size_t output_elems = static_cast<size_t>(signal_len) * batch_size;
    const size_t output_bytes = output_elems * sizeof(float);

    // --- Register Plasma shared-memory buffer as CUDA host memory ---
    float* h_input = register_plasma_buffer(input_plasma_ptr, input_bytes);

    // --- Allocate persistent device memory for the output tensor ---
    // The Rust caller owns this pointer. It must call free_wst_result() when done.
    float* d_output = nullptr;
    cudaError_t err = cudaMalloc(&d_output, output_bytes);
    if (err != cudaSuccess) {
        unregister_plasma_buffer(input_plasma_ptr);
        throw std::runtime_error(
            std::string("cudaMalloc for output tensor failed: ") + cudaGetErrorString(err));
    }

    // --- Execute WST or JTFS pipeline ---
    if (!use_jtfs) {
        // Standard WST path: template-specialised on HopperTag for sm_90,
        // falls back to AmperTag (sm_80) via the TilePolicy mechanism.
        WSTEngine<HopperTag, 8, 16> engine;
        engine.initialise(signal_len, batch_size);

        // H2D transfer via async pinned DMA — overlaps with compute on stream1
        float* d_input = nullptr;
        cudaMalloc(&d_input, input_bytes);
        cudaStream_t h2d_stream;
        cudaStreamCreate(&h2d_stream);
        cudaMemcpyAsync(d_input, h_input, input_bytes, cudaMemcpyHostToDevice, h2d_stream);
        cudaStreamSynchronize(h2d_stream);
        cudaStreamDestroy(h2d_stream);

        engine.forward_pass(d_input, d_output, signal_len, batch_size, depth);
        engine.destroy();
        cudaFree(d_input);
    } else {
        // JTFS path: launches separable 2D convolution on stream0 (time) and
        // stream1 (log-frequency) concurrently.
        JTFSEngine<HopperTag, 8, 16> jtfs_engine(J, Q);

        float* d_input = nullptr;
        cudaMalloc(&d_input, input_bytes);
        cudaStream_t h2d_stream;
        cudaStreamCreate(&h2d_stream);
        cudaMemcpyAsync(d_input, h_input, input_bytes, cudaMemcpyHostToDevice, h2d_stream);
        cudaStreamSynchronize(h2d_stream);
        cudaStreamDestroy(h2d_stream);

        // JTFS forward: streams 0 and 1 run time and frequency convolutions in parallel
        jtfs_engine.forward_jtfs(d_input, d_output, signal_len, batch_size);
        jtfs_engine.destroy();
        cudaFree(d_input);
    }

    // Ensure all device work is complete before returning the output pointer
    cudaDeviceSynchronize();

    // Unregister the Plasma pages — the Plasma store manages their lifetime
    unregister_plasma_buffer(input_plasma_ptr);

    // --- Timing end ---
    auto t_end = std::chrono::high_resolution_clock::now();
    uint64_t elapsed_us = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count());

    return WSTResult {
        /* fingerprint_ptr */ reinterpret_cast<uint64_t>(d_output),
        /* coeff_count     */ output_elems,
        /* exec_time_us    */ elapsed_us
    };
}

// ---------------------------------------------------------------------------
// free_wst_result — Releases the device tensor allocated by run_wst_pipeline.
// The Rust orchestrator must call this after writing the tensor to Plasma.
// ---------------------------------------------------------------------------
void free_wst_result(WSTResult result) {
    if (result.fingerprint_ptr != 0) {
        cudaFree(reinterpret_cast<void*>(result.fingerprint_ptr));
    }
}
