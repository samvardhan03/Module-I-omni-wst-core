#include "memory_staging.cuh"
#include <iostream>
#include <stdexcept>
#include <utility>

MemoryStaging::MemoryStaging(int signal_len, int batch_size, int q) 
    : signal_len_(signal_len), batch_size_(batch_size) {
    
    // At CD-quality (44.1kHz) with Q >= 16, cache consumes 512 MB.
    size_t host_bytes = signal_len * batch_size * sizeof(float);
    
    // Pinned memory for PCIe bypass
    if (cudaMallocHost(&h_input_, host_bytes) != cudaSuccess) {
        throw std::runtime_error("cudaMallocHost failed for h_input_");
    }
    
    if (cudaMallocHost(&h_output_, host_bytes) != cudaSuccess) {
        cudaFreeHost(h_input_);
        throw std::runtime_error("cudaMallocHost failed for h_output_");
    }

    // Device memory double buffering
    cudaMalloc(&d_input_, host_bytes);
    cudaMalloc(&d_input_b_, host_bytes);
    cudaMalloc(&d_output_, host_bytes);
    
    // Create streams for dual-stream double buffering
    cudaStreamCreate(&stream0_); // WST/JTFS forward pass
    cudaStreamCreate(&stream1_); // Pipeline next batch memcpy
}

MemoryStaging::~MemoryStaging() {
    cudaFreeHost(h_input_);
    cudaFreeHost(h_output_);
    cudaFree(d_input_);
    cudaFree(d_input_b_);
    cudaFree(d_output_);
    cudaStreamDestroy(stream0_);
    cudaStreamDestroy(stream1_);
}

void MemoryStaging::load_input(const float* data) {
    size_t host_bytes = signal_len_ * batch_size_ * sizeof(float);
    memcpy(h_input_, data, host_bytes);
}

void MemoryStaging::transfer_to_device_async() {
    size_t bytes = signal_len_ * batch_size_ * sizeof(float);
    // Copy to device buffer b via stream 1 while stream 0 computes
    cudaMemcpyAsync(d_input_b_, h_input_, bytes, cudaMemcpyHostToDevice, stream1_);
    
    // Wait and swap
    cudaStreamSynchronize(stream1_);
    std::swap(d_input_, d_input_b_);
}

void MemoryStaging::transfer_to_host_async() {
    size_t bytes = signal_len_ * batch_size_ * sizeof(float);
    cudaMemcpyAsync(h_output_, d_output_, bytes, cudaMemcpyDeviceToHost, stream0_);
    cudaStreamSynchronize(stream0_);
}

uint64_t MemoryStaging::get_output_uva_handle() const {
    // Return CUdeviceptr representation (64-bit handle)
    return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(d_output_));
}

float* MemoryStaging::get_host_output() const {
    return h_output_;
}

float* MemoryStaging::get_device_input() const {
    return d_input_;
}

float* MemoryStaging::get_device_output() const {
    return d_output_;
}

cudaStream_t MemoryStaging::get_compute_stream() const {
    return stream0_;
}
