# omni-wst-core: GPU-Accelerated Wavelet Scattering & JTFS

**omni-wst-core** is a production-grade, highly optimized C++/CUDA mathematical extension module for Python. It provides extreme-throughput primitives for calculating the **Wavelet Scattering Transform (WST)** and the **Joint Time-Frequency Scattering (JTFS)** transform, designed specifically for formally grounded perceptual fingerprinting and robust signal representation.

By orchestrating dual-stream, double-buffered execution over pinned `cudaMallocHost` memory, this module achieves zero-copy NumPy ingestion and virtually eliminates PCIe bottleneck latency. 

---

## 🔬 Mathematical Formalism

The Wavelet Scattering Transform provides an incredibly rich, deformation-stable representation of high-frequency time-series data. It is constructed through a deep convolutional network architecture where learned filters are replaced by explicit, analytically defined wavelet filter banks.

### 1. The Scattering Cascade

Let $x(u)$ be an input signal. We construct a complex-valued analytic wavelet filter bank $\psi_{\lambda}(u)$ defined by dilations of a mother wavelet, alongside a low-pass scaling function $\phi_J(u)$.

The **zero-order** scattering coefficient is the local average:
$$ S[0]x(u) = x * \phi_J(u) $$

The **first-order** coefficients are obtained by computing the wavelet transform and applying a complex modulus nonlinearity to discard the rapidly varying phase:
$$ S[1]x(u, \lambda_1) = |x * \psi_{\lambda_1}| * \phi_J(u) $$

The **m-order** scattering coefficients are obtained by cascading this convolution and modulus operator $m$ times:
$$ S[m]x(u, \lambda_1, \dots, \lambda_m) = || \dots |x * \psi_{\lambda_1}| * \dots | * \psi_{\lambda_m}| * \phi_J(u) $$

### 2. Parseval Energy Conservation

To prevent the loss of informational energy across the cascade (informational collapse), the filter bank must satisfy a Parseval frame condition. In the Fourier domain, this requires:
$$ |\hat{\phi}_J(\omega)|^2 + \frac{1}{2} \sum_{\lambda} |\hat{\psi}_\lambda(\omega)|^2 \approx 1 $$

By enforcing this condition during the initialization of the `WSTEngine`, we guarantee that the energy of the input signal is perfectly partitioned and conserved across the scattering paths:
$$ \sum_{p} \|S[p]x\|^2 = \|x\|^2 $$

### 3. Adversarial Robustness & Lipschitz Continuity

The fundamental advantage of WST over ad-hoc spectrograms or unconstrained neural networks is **formal deformation stability**. Let $\mathcal{L}_\tau$ be an operator that dilates or translates the signal by a small deformation field $\tau(u)$.

The WST is strictly Lipschitz continuous, meaning that minor deformations result in linearly bounded changes to the fingerprint. The depth-$m$ propagator is bounded by:
$$ \|S[p]x - S[p]y\|_{L^2} \le (\|\psi\|_1)^m \cdot \|x - y\|_{L^2} $$

Because our explicitly constructed Morlet wavelets satisfy $\|\psi\|_1 \le 1$, the Lipschitz constant $L_m$ decays exponentially with depth, strictly bounding the impact of adversarial noise or phase-shifting.

### 4. Joint Time-Frequency Scattering (JTFS)

The standard WST modulus operator $| \cdot |$ destroys critical phase-coupling information between adjacent frequency bands. This can limit the discriminative power for signals characterized by intricate frequency-modulated structures (e.g., chirps in gravitational waves, overlapping vocal formants).

**JTFS** recovers this structure by executing a fully separable 2D convolution across both the temporal axis $t$ and the log-frequency axis $\lambda$:
$$ \Psi_{\mu,l,s}(t,\lambda) = \psi_\mu(t) \cdot \psi_{l,s}(\lambda) $$

`omni-wst-core` computes this massive multidimensional operation concurrently by dispatching parallel CUDA streams (`stream0` for time, `stream1` for frequency) to maximize ALU saturation.

---

## ⚡ Architectural Implementation

This repository is built using an aggressive host-to-device memory architecture to bypass standard PCIe bottlenecks:

- **Template Metaprogramming**: `TilePolicy<ArchTag>` completely eliminates runtime branching, specializing the convolution tile sizes dynamically for Ampere (`TILE=64`) or Hopper (`TILE=128`) architectures at compile time.
- **Zero-Copy FFI**: Utilizing `pybind11` buffer protocols, NumPy memory buffers are seamlessly mapped.
- **UVA & Pinned Memory**: `MemoryStaging` utilizes `cudaMallocHost` explicitly locking RAM. By asynchronously swapping double buffers via `cudaMemcpyAsync`, memory is shuttled to VRAM in the background while the `cuFFT` logic saturates the compute cores.

## 🚀 Quick Start

### Installation

Requires Python 3.10+, CMake 3.18+, and the NVIDIA CUDA Toolkit (v11+ or v12+).

```bash
# Clone and install directly via PEP 517 build
pip install -e .
```

### Usage

```python
import numpy as np
import omni_wst_core as wst

# 1. Initialize configuration for a depth-2 WST
cfg = wst.WSTConfig(J=8, Q=16, depth=2, jtfs=True)

# 2. Simulate 1 second of 44.1kHz audio
signal = np.random.randn(44100).astype(np.float32)

# 3. Generate formally grounded fingerprint
fingerprint = wst.fingerprint(signal, cfg)

print(f"Fingerprint Dimensions: {fingerprint.shape}")
```

## ⚖️ License
The mathematical primitives within `omni-wst-core` are licensed under the **Apache License 2.0** for free open-source academic and research usage.

**Commercial Deployment:** Production deployment of OmniPulse modules requires an Enterprise SaaS agreement. Academic and government institutions qualify for immediate enterprise waivers. Please refer to `COMMERCIAL_LICENSE.md`.
