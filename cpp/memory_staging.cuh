#ifndef MEMORY_STAGING_CUH
#define MEMORY_STAGING_CUH

#include <cstdint>
#include <cuda_runtime.h>

class MemoryStaging {
public:
    MemoryStaging(int signal_len, int batch_size, int q);
    ~MemoryStaging();

    void load_input(const float* data);
    void transfer_to_device_async();
    void transfer_to_host_async();
    
    uint64_t get_output_uva_handle() const;
    float* get_host_output() const;
    float* get_device_input() const;
    float* get_device_output() const;
    cudaStream_t get_compute_stream() const;

private:
    float* h_input_;
    float* h_output_;
    
    float* d_input_;
    float* d_input_b_;
    float* d_output_;
    
    cudaStream_t stream0_;
    cudaStream_t stream1_;
    
    int signal_len_;
    int batch_size_;
};

#endif // MEMORY_STAGING_CUH
