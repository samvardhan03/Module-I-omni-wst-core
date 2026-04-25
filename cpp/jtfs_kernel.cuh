#ifndef JTFS_KERNEL_CUH
#define JTFS_KERNEL_CUH

#include "wst_kernel.cuh"

// Config parameter struct for JTFS
struct JTFSConfig {
    int J_fr;
    int Q_fr;
};

// Phase 1 Kernel (Time-axis convolution wrapper)
__global__ void time_conv_kernel(Complex* U1, Complex* psi_mu, Complex* out, int time_len, int freq_len) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int lambda = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (t < time_len && lambda < freq_len) {
        int idx = lambda * time_len + t;
        // pointwise multiplication across time
        Complex res;
        res.x = U1[idx].x * psi_mu[t].x - U1[idx].y * psi_mu[t].y;
        res.y = U1[idx].x * psi_mu[t].y + U1[idx].y * psi_mu[t].x;
        out[idx] = res;
    }
}

// Phase 2 Kernel (Log-frequency convolution wrapper)
__global__ void freq_conv_kernel(Complex* intermediate, Complex* psi_l_s, Complex* out, int time_len, int freq_len) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int lambda = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (t < time_len && lambda < freq_len) {
        int idx = lambda * time_len + t;
        // pointwise multiplication across frequency
        Complex res;
        res.x = intermediate[idx].x * psi_l_s[lambda].x - intermediate[idx].y * psi_l_s[lambda].y;
        res.y = intermediate[idx].x * psi_l_s[lambda].y + intermediate[idx].y * psi_l_s[lambda].x;
        out[idx] = res;
    }
}

template<typename ArchTag, int J, int Q, int J_fr>
class JTFSEngine : public WSTEngine<ArchTag, J, Q> {
public:
    JTFSEngine(const WSTConfig& cfg, const JTFSConfig& jtfs_cfg) 
        : WSTEngine<ArchTag, J, Q>(cfg), jtfs_config_(jtfs_cfg) {
        // Pre-allocate d_freq_filter_bank
        int lambda_in = J * Q;
        int num_freq_filters = J_fr * jtfs_cfg.Q_fr;
        size_t bytes = num_freq_filters * lambda_in * sizeof(Complex);
        
        cudaError_t err = cudaMalloc(&d_freq_filter_bank_, bytes);
        if (err != cudaSuccess) {
            throw std::runtime_error("Failed to allocate d_freq_filter_bank");
        }
        
        // Ensure zero initialization
        cudaMemset(d_freq_filter_bank_, 0, bytes);
        
        // Create streams for two-phase execution
        cudaStreamCreate(&stream0_);
        cudaStreamCreate(&stream1_);
    }
    
    ~JTFSEngine() override {
        cudaFree(d_freq_filter_bank_);
        cudaStreamDestroy(stream0_);
        cudaStreamDestroy(stream1_);
    }

    // Launch separable 2D wavelet convolution
    void launch_jtfs_pipeline(Complex* d_U1, Complex* d_psi_mu, Complex* d_psi_l_s, Complex* d_intermediate, Complex* d_out) {
        int time_len = this->config_.signal_len;
        int freq_len = J * Q;
        
        dim3 blockDim(16, 16);
        dim3 gridDim((time_len + 15) / 16, (freq_len + 15) / 16);
        
        // Phase 1: Time-axis convolution on stream0
        time_conv_kernel<<<gridDim, blockDim, 0, stream0_>>>(d_U1, d_psi_mu, d_intermediate, time_len, freq_len);
        
        // Barrier: Wait for Phase 1 to finish
        cudaStreamSynchronize(stream0_);
        
        // Phase 2: Log-frequency convolution on stream1
        freq_conv_kernel<<<gridDim, blockDim, 0, stream1_>>>(d_intermediate, d_psi_l_s, d_out, time_len, freq_len);
        
        // Barrier: Wait for Phase 2 to finish
        cudaStreamSynchronize(stream1_);
    }

private:
    JTFSConfig jtfs_config_;
    Complex* d_freq_filter_bank_;
    cudaStream_t stream0_;
    cudaStream_t stream1_;
};

#endif // JTFS_KERNEL_CUH
