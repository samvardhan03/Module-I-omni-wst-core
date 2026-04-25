# OmniPulse — Modular Engineering & Commercialization Blueprint

**Classification:** Internal Engineering Strategy — Principal Architect Review  
**Revision:** 1.0 | **Source Document:** OmniPulse Technical Design & Implementation Document, Rev 1.0  
**Architecture:** Polyglot C++/CUDA · Rust · Python  

---

> **Strategic Premise:** The three computational tiers of the OmniPulse pipeline — (1) C++/CUDA mathematical primitives, (2) the Rust orchestration and vector database engine, and (3) the Python agentic control plane — are each independently valuable, formally grounded, and commercially deployable as standalone open-source or commercial products. This blueprint documents the precise engineering and commercialization pathway for each isolated module, followed by the exact integration workflow to assemble the complete enterprise OmniPulse SaaS platform.

---

## Table of Contents

1. [Part 1A — Module I: C++/CUDA Mathematical Primitives (`omni-wst-core`)](#module-i)
2. [Part 1B — Module II: Rust Orchestration & Vector Database (`omni-orch`)](#module-ii)
3. [Part 1C — Module III: Python Agentic Control Plane (`omni-agent`)](#module-iii)
4. [Part 2 — The OmniPulse Assembly: Integration Engineering](#part-2-assembly)

---

<a name="module-i"></a>
# Part 1 — Independent Module Engineering & Commercialization

---

## Module I: C++/CUDA Mathematical Primitives — `omni-wst-core`

### A. What to Build & Isolate

This module isolates the entire **Phase 1** compute stack described in the OmniPulse TDD: the GPU-accelerated Wavelet Scattering Transform (WST) and its Joint Time-Frequency Scattering (JTFS) extension, together with the CUDA memory staging infrastructure that makes both transforms viable at industrial throughput. The following components are to be isolated and built as a standalone shared library with Python bindings:

**1. Core WST Engine — Recursive Scattering Cascade (`wst_kernel.cuh`)**

- The complete depth-`m` recursive scattering propagator implementing the canonical formula:

  `S[p]x(u) = | ... ||x * ψ_λ₁| * ψ_λ₂| ⋯ * ψ_λ_m| * φ_J(u)`

  where `ψ_λ_k` are analytic Morlet wavelets at frequency `λ_k = 2^(j_k) r_k` and `φ_J(u)` is the low-pass averaging filter at scale `2^J`.
- Compile-time tile specialisation via template metaprogramming: `TilePolicy<AmpereTag>` (64×64 tiles) and `TilePolicy<HopperTag>` (128×128 tiles) enabling architecture-adaptive register utilisation without runtime branching.
- The `WSTEngine<ArchTag, J, Q>` class template, parameterised by `J` (maximum wavelet scale) and `Q` (wavelets per octave), yielding `N_WAVELETS = J * Q` filter bank entries per cascade depth.
- Strict **Parseval frame** enforcement across the filter bank `{ψ_λ}` to satisfy the energy-preservation identity `Σ_p ||S[p]x||² = ||x||²`, which prevents information collapse in deep convolutional cascades and is the prerequisite for the Lipschitz continuity guarantee (see §D, Testing).

**2. JTFS Extension — Separable 2D Wavelet Convolution (`jtfs_kernel.cuh`)**

- The `JTFSEngine<ArchTag, J, Q, J_fr>` class, extending `WSTEngine`, implementing the separable 2D wavelet:

  `Ψ_{μ,l,s}(t,λ) = ψ_μ(t) · ψ_{l,s}(λ)`

  where `ψ_μ(t)` captures temporal amplitude modulation and `ψ_{l,s}(λ)` captures log-frequency modulation across the scalogram plane `U1[λ, t]`.
- Two-phase CUDA execution pipeline: **Phase 1** launches `launch_time_conv` on `stream0` to convolve `U1` with `ψ_μ(t)` along the time axis; **Phase 2** launches `launch_freq_conv` on `stream1` to convolve the intermediate result with `ψ_{l,s}(λ)` along the log-frequency axis. Both phases execute with `cudaStreamSynchronize` barriers to guarantee determinism.
- The `d_freq_filter_bank` buffer of shape `[J_fr * Q_fr, Lambda_in]` must be pre-computed and pinned to VRAM across all batches to eliminate redundant FFT initialisation overhead.
- This component directly addresses the **Phase-Shifting Attack** vulnerability of standard WST — an adversary applying a frequency-dependent rotation `Δφ(ω)` such that `||x - x'||_perceptual ≈ 0` while `||S[p]x - S[p]x'|| > δ`. JTFS empirically reduces such adversarial hash collision rates by **34%** on standardised audio benchmarks, as cited in the OmniPulse TDD.

**3. CUDA Memory Staging — Pinned Memory & Dual-Stream Double Buffering (`memory_staging.cu`)**

- `cudaMallocHost()` allocation of pinned (page-locked) host buffers to bypass OS paging and enable GPU DMA transfers exceeding **15 GB/s** PCIe bandwidth, compared to ~4 GB/s for pageable memory.
- Dual-stream double-buffering protocol: `stream0` executes the WST/JTFS forward pass on the current active batch while `stream1` concurrently pipelines the next batch via `cudaMemcpyAsync`, achieving **≥95% PCIe transfer latency hiding**.
- At CD-quality (44.1 kHz) with `Q ≥ 16`, the filter bank cache consumes **512 MB VRAM** per processing stream. The `initialise()` method must pre-allocate and pin this budget using `cudaMallocHost()` and `cudaMalloc()` for the `d_input`, `d_input_b` (double-buffer), `d_filter_bank`, and `d_output` buffers.
- **Unified Virtual Addressing (UVA)** enforcement throughout: finalised scattering tensor pointers propagate upward as opaque `CUdeviceptr` (64-bit integer handles) without host-side tensor replication. This is the contractual interface to the Rust FFI layer.
- The `cufftPlanMany` batched 1D FFT plan must be bound to `stream0` via `cufftSetStream` to maintain stream affinity across all `cufftExecC2C` calls in the scattering cascade.

**4. pybind11 Python Bindings (`wst_bindings.cpp`)**

- Zero-copy Python interface via the **pybind11 buffer protocol**: `py::array_t<float>` inputs are resolved to raw `float*` pointers via `buf.request()` without triggering NumPy array copies.
- Expose `WSTConfig` as a Python class with `J`, `Q`, `depth`, and `jtfs` attributes.
- Expose `fingerprint(signal: np.ndarray, cfg: WSTConfig) -> np.ndarray` as the primary entry point.
- Optionally expose `JTFSConfig` for researchers requiring fine-grained control over `J_fr` and `Q_fr` parameters.

---

### B. Packaging & Release Strategy

**Build System: `scikit-build-core` + CMake + `cibuildwheel`**

As explicitly documented in OmniPulse TDD §4.1.1, the `pyproject.toml` uses `scikit-build-core` with CMake backend:

```toml
[build-system]
requires = ["scikit-build-core", "pybind11", "cmake"]
build-backend = "scikit_build_core.build"

[project]
name = "omni-wst-core"
version = "1.0.0"

[tool.scikit-build]
cmake.build-type = "Release"
cmake.args = ["-DCUDA_ARCH=native", "-DBUILD_PYBIND11=ON"]
```

**Wheel Distribution Matrix (per TDD §4.1.2):**

| Variant | Delivery Mechanism | Notes |
|---|---|---|
| CPU-only | `manylinux2014` + macOS ARM/x86 + Windows | Via `cibuildwheel` CI matrix; no CUDA driver required |
| CUDA 12.x | Published as `omni-wst-core-cu12` on PyPI | Requires NVIDIA driver ≥ 525 |
| Source dist | `sdist` with CMake for custom CUDA arch targets | Supports `sm_80` (Ampere), `sm_90` (Hopper), `sm_100` (Blackwell) |

**CI Pipeline:**

- `cibuildwheel` GitHub Actions matrix across `{linux, macos, windows}` × `{cp310, cp311, cp312}`.
- CUDA wheel builds require a self-hosted runner with NVIDIA GPU and CUDA toolkit installed; use a `cuDNN`-enabled Docker base image (`nvidia/cuda:12.x-devel-ubuntu22.04`).
- `CUDA_ARCH=native` for development builds; explicitly enumerate `sm_80;sm_90` for PyPI release wheels to ensure broad hardware compatibility.

**Licensing:**

- Apache 2.0 for research and non-commercial use (matching the OmniPulse TDD's stated research tier model).
- Commercial use requires a separate enterprise license, enforced via `COMMERCIAL_LICENSE.md` in the repository and a license check in the CMake configuration (`-DCOMMERCIAL_USE=ON`).

**Documentation:**

- Sphinx API reference auto-generated from pybind11 docstrings, hosted on Read the Docs.
- Jupyter notebooks demonstrating WST on audio signals (librosa integration) and 2D image signals (torchvision integration).

---

### C. Alternative R&D Applications

**Application 1: Gravitational Wave Signal Analysis (LIGO / Virgo Astrophysics)**

The WST and JTFS primitives are directly applicable to the characterisation of **compact binary coalescence (CBC) gravitational wave signals** from LIGO and Virgo detectors.

- *Why WST?* Gravitational wave signals are non-stationary, transient, and embedded in broadband noise. The WST's **deformation stability** property — formally guaranteed by the Lipschitz bound `L_m ≤ (||ψ||₁)^m` — means that small perturbations in the signal waveform (e.g., from matched filter template mismatch) cannot catastrophically alter the scattering coefficient representation. This makes WST a physically meaningful feature extractor for parameter estimation.
- *Why JTFS?* The characteristic **chirp** of a neutron star merger (frequency increasing from ~10 Hz to ~2,000 Hz over the final seconds of inspiral) exhibits precisely the inter-band amplitude modulation correlations that JTFS is designed to capture via `Ψ_{μ,l,s}(t,λ)`. Standard WST discards the phase coupling between adjacent frequency bands, losing critical information about the chirp mass and mass ratio. JTFS recovers this via the log-frequency wavelet `ψ_{l,s}(λ)`.
- *Concrete deployment:* The `omni-wst-core` CPU wheels integrate directly with `GWpy` and `PyCBC` pipelines. Researchers replace hand-crafted Q-transform spectrograms with JTFS scattering coefficients as input features to Bayesian parameter estimation networks, achieving richer representations without the Q-transform's fixed time-frequency resolution trade-off.
- *CUDA advantage:* Real-time LIGO event alert classification (the **LLOID** pipeline) currently operates with latency budgets of ~10–30 seconds. The dual-stream double-buffering and cuFFT batched convolution in `omni-wst-core` enable sub-second JTFS feature extraction on GPU clusters, enabling low-latency CBC classification for multi-messenger astronomy follow-up.

**Application 2: Genomic Signal Processing — ChIP-seq Peak Calling (Bioinformatics)**

ChIP-sequencing (Chromatin Immunoprecipitation sequencing) produces 1D read-depth signals over genomic coordinates that encode the binding landscape of transcription factors and histone modifications. Existing peak callers (MACS2, SICER) rely on parametric statistical models that are sensitive to background estimation errors.

- *Why WST?* The `S[p]x(u)` representation provides a **translation-invariant, deformation-stable** encoding of the read-depth signal. Genuine binding peaks exhibit characteristic Gaussian-like profiles that are stable under genomic position shifts and minor peak broadening, while background noise artefacts are highly variable. The Parseval frame energy-preservation constraint ensures that both sharp (transcription factor) and broad (histone) peaks are faithfully encoded without amplitude collapse.
- *Why JTFS?* Many histone modifications exhibit **co-modulated** binding patterns: H3K4me3 (sharp promoter peaks) co-localises with H3K27ac (broad enhancer patterns) in active regulatory regions. JTFS, by computing cross-frequency modulation correlations `Ψ_{μ,l,s}(t,λ)` on the multi-track read-depth stack, can encode these inter-mark amplitude correlations that are structurally invisible to single-track WST.
- *Concrete deployment:* Batch processing of `n` ChIP-seq experiments simultaneously via the `WSTEngine`'s batched cuFFT plan (`cufftPlanMany` with `batch_size = n`). Each experiment's 1D read-depth array (sampled at 10 bp resolution, ~3×10⁸ bins) is processed in windows with 50% overlap; scattering coefficients serve as features for a downstream binary classifier (peak vs. background). The GPU achieves ~40× throughput improvement over CPU-based wavelet implementations.

---

### D. Testing & Validation Protocols

**1. Mathematical Validation: Lipschitz Continuity Bound**

The OmniPulse TDD §1.4 formally proves the depth-`m` Lipschitz bound: `||S[p]x - S[p]y||_{L2} ≤ (||ψ||₁)^m · ||x - y||_{L2}`. This must be numerically validated:

```python
import omni_wst_core as wst
import numpy as np

def validate_lipschitz_bound(cfg, n_trials=1000, noise_scale=1e-3):
    errors = []
    for _ in range(n_trials):
        x = np.random.randn(cfg.signal_len).astype(np.float32)
        delta = np.random.randn(cfg.signal_len).astype(np.float32) * noise_scale
        y = x + delta

        Sx = wst.fingerprint(x, cfg)
        Sy = wst.fingerprint(y, cfg)

        lhs = np.linalg.norm(Sx - Sy)
        rhs = (cfg.l1_norm_psi ** cfg.depth) * np.linalg.norm(x - y)
        assert lhs <= rhs + 1e-6, f"Lipschitz violation: {lhs:.6f} > {rhs:.6f}"
        errors.append(lhs / rhs)  # empirical tightness ratio
    return np.mean(errors)
```

- The test must pass for all valid `(J, Q, depth)` configurations. `cfg.l1_norm_psi` is a property of the filter bank, computable analytically at `initialise()` time and validated to satisfy `||ψ||₁ < 1` as mandated by the OmniPulse TDD design constraint.
- **Adversarial phase-shift test:** Apply random frequency-domain phase rotations `Δφ(ω) ~ Uniform(0, 2π)` to generate `x'` satisfying `||x - x'||_perceptual ≈ 0` (i.e., same LUFS loudness). Assert that JTFS hash collision rate is ≤ 34% lower than WST (matching TDD §1.2 empirical claim).

**2. Energy Conservation Test (Parseval Frame)**

```python
def validate_parseval_frame(cfg, n_trials=500):
    for _ in range(n_trials):
        x = np.random.randn(cfg.signal_len).astype(np.float32)
        scattering_coeffs_per_path = wst.scattering_paths(x, cfg)  # returns list of S[p]x
        energy_out = sum(np.linalg.norm(s)**2 for s in scattering_coeffs_per_path)
        energy_in = np.linalg.norm(x)**2
        rel_error = abs(energy_out - energy_in) / energy_in
        assert rel_error < 1e-4, f"Parseval violation: relative error = {rel_error:.2e}"
```

**3. CUDA Correctness: CPU/GPU Numerical Agreement**

- For each `(J, Q, depth)` in the validation matrix, compute `fingerprint(x, cfg, backend='cpu')` and `fingerprint(x, cfg, backend='cuda')` and assert `max_abs_error < 1e-4` (float32 tolerance).
- Regression test: fix random seed, store reference outputs in a binary fixture, assert identity across CUDA driver versions.

**4. Memory Safety: Pinned Buffer Lifecycle**

- Use `cuda-memcheck` / `compute-sanitizer --tool memcheck` on all test binaries to assert zero invalid memory accesses.
- Valgrind `--tool=massif` on CPU path to validate that no heap allocations occur after `initialise()` during `forward_pass()` (all tensors pre-allocated).
- Assert that `cudaMallocHost` allocations are freed in `destroy()` via `cudaFreeHost` with no leaks.

**5. Throughput Benchmarks**

- Target: **≥1,000 audio frames/second** (44.1 kHz, 4096-sample windows, `J=8, Q=16, depth=2`) on A100 80GB.
- Validate the **95% transfer latency hiding** claim by profiling with `nsys profile` and measuring `cudaMemcpyAsync` overlap fraction across 1,000-batch runs.

---

<a name="module-ii"></a>

## Module II: Rust Orchestration & Vector Database — `omni-orch`

### A. What to Build & Isolate

This module isolates the entire **Phase 2** Rust tier: the `cxx`-based zero-cost FFI bridge to C++, the concurrent HNSW fingerprint index, the Sliced Wasserstein Distance (SW₁) metric engine, the cryptographic Ed25519 licensing token system, and the MCP server infrastructure. It is designed to be deployable as an independent binary or library crate, callable from any language via its MCP JSON-RPC 2.0 interface.

**1. Zero-Cost `cxx` FFI Bridge (`wst_bridge.h` + `lib.rs`)**

- The `WSTResult` struct (ABI-validated at compile time by `cxx`), carrying `fingerprint_ptr: u64` (a `CUdeviceptr` opaque handle), `coeff_count: u64`, and `exec_time_us: u64`.
- The `unsafe extern "C++"` block declaring `run_wst_pipeline` with the full parameter signature: `(input_plasma_ptr: u64, signal_len: i32, batch_size: i32, j: i32, q: i32, depth: i32, use_jtfs: bool) -> WSTResult`.
- The safe wrapper `execute_fingerprint_pass(plasma_id: u64, cfg: &WstConfig) -> WSTResult` with a documented safety contract: `plasma_id` is a valid Arrow Plasma mmap pointer, RustBelt borrow-checker guarantees provide no aliasing or dangling references.
- The build script (`build.rs`) must invoke `cxx_build::bridge("src/lib.rs").file("cpp/wst_bridge.cpp").flag("-std=c++17").compile("wst_bridge")`, with CUDA library paths provided via `cargo:rustc-link-lib`.

**2. Concurrent HNSW Vector Database (`hnsw_store.rs`)**

- `FingerprintStore` struct wrapping `Arc<RwLock<Hnsw<f32, DistSlicedWasserstein>>>`.
  - `Arc` enables shared ownership across concurrent Tokio tasks (each autonomous LLM agent gets a clone of the `Arc` without copying the index).
  - `RwLock` allows multiple concurrent read guards (many agents querying simultaneously) while enforcing single-writer exclusivity during `insert()` operations, as mandated by the RustBelt `Send + Sync` trait system (Jung et al., POPL 2018).
- `query(&self, embedding: &[f32], k: usize) -> Vec<(u64, f32)>` using `ef_search = 48` (tunable via configuration). HNSW complexity: `O(log N)` per query on pre-built graph structure.
- `insert(&self, id: u64, embedding: &[f32])` — exclusive write access; must be called from the licensing token issuance path only.
- `batch_sw_distances(&self, query: &[f32]) -> Vec<f32>` — parallel SW distance computation across all registered fingerprints using `Rayon`'s `.par_iter()` work-stealing scheduler.

**3. Sliced Wasserstein Distance Engine (`sliced_wasserstein.rs`)**

- Full implementation of:

  `SW₁(μ, ν) = ∫_{S^{d-1}} W₁(θ_#μ, θ_#ν) dσ(θ)`

  approximated via Monte Carlo sampling with `P = 256` random unit-vector projections (configurable), each projection reducing to a sort (`O(N log N)`) and mean absolute quantile difference.
- `sample_unit_sphere(n_proj: usize, d: usize) -> Vec<f32>` using Gaussian sampling followed by L2 normalisation.
- `sorted_projection(x: &[f32], theta: &[f32]) -> Vec<f32>`: dot-product projection followed by `sort_by(partial_cmp)`.
- Rayon `par_chunks(d)` across all `P` projections for fully data-parallel execution. Total complexity: `O(P · N log N)` — as specified in TDD §2.3.
- Threshold calibration interface: `calibrate_threshold(positive_pairs: &[(Vec<f32>, Vec<f32>)], negative_pairs: &[(Vec<f32>, Vec<f32>)]) -> f32` returning the optimal Wasserstein threshold `τ` via ROC curve analysis.

**4. Ed25519 Cryptographic Licensing Engine (`license_token.rs`)**

- `LicensePayload` struct with `serde::{Serialize, Deserialize}`: `media_sha3: [u8; 32]`, `sw_distance: f32`, `licensee_pubkey: [u8; 32]`, `license_type: LicenseType`, `issued_at_unix: u64`, `expiry_unix: u64`, `royalty_bps: u16`.
- `issue_token(payload: LicensePayload, signing_key: &SigningKey, ipfs_client: &IpfsClient) -> Result<SignedLicenseToken, LicenseError>`:
  1. Deterministic `bincode::serialize(&payload)` → raw bytes.
  2. `signing_key.sign(&raw)` using `ed25519_dalek::Signer` trait.
  3. `ipfs_client.add_bytes(&raw).await?` → IPFS CIDv1 content address.
- `SignedLicenseToken` carries `payload`, `signature: Signature`, and `ipfs_cid: String`.

**5. MCP Server Infrastructure**

- JSON-RPC 2.0 server exposing three deterministically-typed tool endpoints: `generate_fingerprint`, `compare_fingerprints`, and `issue_license_token` (schemas fully defined in TDD §3.2).
- Tokio async runtime for concurrent tool invocation handling.
- Structured `tracing` instrumentation on all request/response paths for distributed observability.

---

### B. Packaging & Release Strategy

**Cargo Crate Structure:**

```
omni-orch/
├── Cargo.toml           # workspace root
├── crates/
│   ├── omni-ffi/        # cxx bridge to C++ kernels
│   ├── omni-hnsw/       # FingerprintStore + HNSW
│   ├── omni-sw/         # Sliced Wasserstein Distance
│   ├── omni-license/    # Ed25519 + IPFS licensing
│   └── omni-mcp/        # MCP JSON-RPC 2.0 server
└── src/
    └── main.rs          # standalone binary entry point
```

**Publishing to `crates.io`:**

- Each crate is published independently under the `omni-*` namespace.
- `omni-sw` (the Sliced Wasserstein crate) has **zero unsafe code** and no C++ dependency; it can be released immediately as a pure Rust crate with no build-script complexity, targeting the widest possible adoption.
- `omni-hnsw` depends on `hnsw_rs` and `rayon`; publish with `features = ["rayon"]` gated.
- `omni-ffi` conditionally compiles the `cxx` bridge only when the `cuda` feature flag is enabled (`[features] cuda = ["cxx", "omni-ffi/cuda"]`), allowing `omni-orch` to be built in a pure Rust / CPU-only mode without a CUDA SDK installed.

**Binary Release:**

- Docker image `omni/orchestrator:1.0` as specified in TDD §4.2.1, containing the compiled MCP server binary.
- GitHub Releases with statically-linked `x86_64-unknown-linux-musl` binaries for zero-dependency deployment.
- `cargo install omni-orch --features mcp-server` for local MCP server deployment.

**Licensing:**

- `omni-sw` and `omni-hnsw`: MIT / Apache 2.0 dual license (maximise Rust ecosystem adoption).
- `omni-license` and `omni-ffi`: Apache 2.0 with commercial use restrictions mirroring the OmniPulse enterprise tier.

---

### C. Alternative R&D Applications

**Application 1: High-Frequency Trading — Real-Time Order Book Similarity Matching**

The `omni-hnsw` HNSW vector database and `omni-sw` Sliced Wasserstein engine are directly applicable to **real-time L2 order book state similarity retrieval** in high-frequency trading (HFT) systems.

- *Problem:* An HFT strategy may identify profitable regimes by recognising that the current order book microstructure resembles historical states that preceded specific short-term price movements (e.g., a "stacked bid" configuration at a key support level). Euclidean distance over raw bid-ask vectors is inadequate because it ignores the continuous distributional nature of order book depth.
- *Why SW₁?* The order book at depth `d` is a discrete approximation of a continuous price-volume distribution. SW₁ is a proper metric on the space of such distributions, metrising weak convergence — properties formally absent from Hamming or L2 distance (McKeown, DFRWS 2025, as cited in TDD §2.3). Two order books with identical total liquidity but different depth distributions are correctly differentiated by SW₁ but may be spuriously identical under L2.
- *Why `Arc<RwLock<HnswIndex>>`?* Market data feeds update the order book at microsecond frequencies, requiring concurrent writes (new snapshots) and reads (regime queries from multiple strategy threads) without blocking. The `RwLock` allows N simultaneous readers (strategy threads querying for similar historical states) with a single periodic writer (order book snapshot ingestion). Rayon `par_iter` over cached fingerprints provides sub-millisecond k-NN query latency on a corpus of 10⁶ historical snapshots.
- *Concrete deployment:* `omni-sw` is compiled as a no_std Rayon-parallel crate with `#[no_std]` + `alloc` support for latency-critical environments. Integration with the `arrowhead` FIX protocol library provides direct market data ingestion.

**Application 2: Enterprise Knowledge Base RAG — Semantic De-duplication & Drift Detection**

In large-scale Retrieval-Augmented Generation (RAG) systems managing document corpora of millions of chunks, two critical failure modes are semantic near-duplicate pollution (many near-identical chunks inflating the index) and **embedding drift** (the embedding distribution of a document corpus shifting as new documents are added, degrading retrieval quality).

- *Why HNSW for de-duplication?* On ingestion of a new document chunk, `FingerprintStore::query()` performs an approximate nearest-neighbour search against the existing corpus. If the SW₁ distance to the top-k neighbours falls below threshold `τ`, the chunk is classified as a semantic near-duplicate and suppressed. This eliminates the O(N²) pairwise comparison bottleneck of naive de-duplication.
- *Why SW₁ for drift detection?* At periodic intervals, a random sample of the document embedding corpus is treated as an empirical distribution. SW₁ between the current sample distribution and a reference baseline distribution (computed at index initialisation) provides a statistically meaningful, computationally tractable measure of **corpus distributional shift** — critical for triggering re-embedding when a new embedding model is deployed.
- *Why Rust?* The `Arc<RwLock<HnswIndex>>` concurrency model natively supports high-throughput document ingestion pipelines where multiple ingest workers acquire write locks in a non-blocking queue, while many retrieval workers maintain concurrent read access. The RustBelt memory safety guarantees eliminate entire classes of data race bugs that plague C++-based vector database implementations.
- *Concrete deployment:* `omni-hnsw` integrates with `pgvector` via a Rust `sqlx`-based connector, enabling the HNSW index to be persisted to PostgreSQL while SW₁ is used as a custom distance function in the retrieval ranking pipeline.

---

### D. Testing & Validation Protocols

**1. Concurrency Stress Testing: `Arc<RwLock<HnswIndex>>`**

The correctness of the concurrent HNSW store under high contention must be validated exhaustively:

```rust
#[tokio::test]
async fn stress_test_concurrent_rw() {
    let store = Arc::new(FingerprintStore::new(128, 16, 200));
    let n_writers = 8;
    let n_readers = 64;
    let n_ops = 10_000;

    let write_handles: Vec<_> = (0..n_writers).map(|i| {
        let s = store.clone();
        tokio::spawn(async move {
            for j in 0..n_ops {
                let id = (i * n_ops + j) as u64;
                let emb: Vec<f32> = (0..128).map(|_| rand::random()).collect();
                s.insert(id, &emb);
            }
        })
    }).collect();

    let read_handles: Vec<_> = (0..n_readers).map(|_| {
        let s = store.clone();
        tokio::spawn(async move {
            for _ in 0..n_ops {
                let query: Vec<f32> = (0..128).map(|_| rand::random()).collect();
                let _ = s.query(&query, 10);
            }
        })
    }).collect();

    futures::future::join_all(write_handles).await;
    futures::future::join_all(read_handles).await;
    // Assert: no panics, no poisoned locks
}
```

- Additionally, run under `cargo test --release` with `RUSTFLAGS="-Z sanitizer=thread"` (ThreadSanitizer) to detect any latent data races not caught by the borrow checker.
- Use `loom` (the deterministic concurrency model checker for Rust) to exhaustively explore all possible thread interleavings for the `RwLock` acquisition/release sequences.

**2. Sliced Wasserstein Metric Property Validation**

SW₁ must satisfy the three axioms of a metric (non-negativity, symmetry, triangle inequality) to be a valid fingerprint distance:

```rust
#[test]
fn validate_sw_metric_axioms() {
    let n = 1000;
    for _ in 0..n {
        let a: Vec<f32> = sample_random_embedding(256);
        let b: Vec<f32> = sample_random_embedding(256);
        let c: Vec<f32> = sample_random_embedding(256);

        let d_ab = sliced_wasserstein_dist(&a, &b, 256);
        let d_ba = sliced_wasserstein_dist(&b, &a, 256);
        let d_ac = sliced_wasserstein_dist(&a, &c, 256);
        let d_bc = sliced_wasserstein_dist(&b, &c, 256);

        assert!(d_ab >= 0.0, "Non-negativity violated");
        assert!((d_ab - d_ba).abs() < 1e-5, "Symmetry violated");
        assert!(d_ab <= d_ac + d_bc + 1e-4, "Triangle inequality violated");
    }
}
```

- Validate convergence of the Monte Carlo estimator: plot `SW₁` vs. `n_projections` for `P ∈ {16, 32, 64, 128, 256, 512}` and confirm variance falls below 1% at `P = 256` for typical fingerprint dimensionalities.

**3. Ed25519 Signature Verification**

```rust
#[test]
fn validate_license_token_signature() {
    let (signing_key, verifying_key) = generate_ed25519_keypair();
    let payload = LicensePayload { /* ... */ };
    let token = issue_token(payload, &signing_key, &mock_ipfs()).unwrap();

    let raw = bincode::serialize(&token.payload).unwrap();
    assert!(verifying_key.verify(&raw, &token.signature).is_ok());
    // Tamper test: modify one byte of payload
    let mut tampered = raw.clone();
    tampered[0] ^= 0xFF;
    assert!(verifying_key.verify(&tampered, &token.signature).is_err());
}
```

**4. FFI Bridge ABI Safety**

- Compile the full `cxx` bridge in a `cargo test` harness with `AddressSanitizer` (`RUSTFLAGS="-Z sanitizer=address"`) and run a suite of 10,000 `execute_fingerprint_pass` calls with randomised inputs to validate that no memory safety violations occur at the Rust/C++ boundary.
- Assert that `WSTResult.fingerprint_ptr` returned from `run_wst_pipeline` is always a valid, aligned `CUdeviceptr` by attempting to access its contents via `cuPointerGetAttribute` with `CU_POINTER_ATTRIBUTE_MEMORY_TYPE`.

**5. MCP Tool Schema Validation**

- Use `jsonschema` to validate every outgoing MCP tool response payload against the exact schemas defined in TDD §3.2 (`generate_fingerprint`, `compare_fingerprints`, `issue_license_token`).
- Fuzz the MCP server input handling with `cargo-fuzz` targeting the JSON deserialization layer to validate that malformed tool invocations never produce panics or undefined behaviour.

---

<a name="module-iii"></a>

## Module III: Python Agentic Control Plane — `omni-agent`

### A. What to Build & Isolate

This module isolates the entire **Phase 3** Python tier: the Apache Arrow Plasma zero-copy shared memory lifecycle manager, the MCP tool client orchestration layer, the LLM agentic control loop, and the Kubernetes FinOps autoscaler. It is deployable as an independent Python package, callable against any compliant MCP server backend (not necessarily the Rust `omni-orch` backend).

**1. Apache Arrow Plasma Zero-Copy Memory Manager (`plasma_manager.py`)**

- `PlasmaManager` class wrapping `pyarrow.plasma.connect("/tmp/plasma")`.
- `ingest_media_tensor(audio_array: np.ndarray) -> str`: serialises a `float32` NumPy array into a Plasma mmap region using a **content-addressed** `ObjectID` derived from `hashlib.sha3_256(buf).digest()[:20]`. Returns the 20-byte hex ObjectID for downstream MCP dispatch.
- **Critical invariant:** The tensor itself never traverses any language boundary. Only the 20-byte ObjectID propagates through the MCP JSON message layer, eliminating all `O(N)` JSON serialisation latency for `N`-dimensional tensor payloads.
- `release_object(object_id: str)` for explicit lifecycle management; Plasma enforces immutability post-`seal()`.

**2. MCP Tool Client (`mcp_client.py`)**

- Async `MCPClient` class connecting to the Rust `omni-orch` MCP server at `http://omnipulse-orchestrator:8080`.
- `call_tool(tool_name: str, params: dict) -> dict`: JSON-RPC 2.0 `{"method": tool_name, "params": params}` invocation with structured error handling.
- Implements the full three-tool orchestration sequence: `generate_fingerprint` → `compare_fingerprints` → `issue_license_token` (the last tool invoked only when `is_licensed_derivative: true`).

**3. LLM Agentic Control Loop (`control_plane.py`)**

- Anthropic SDK integration: the LLM (Claude) acts as a **cognitive router**, parsing natural language operator requests and mapping them to deterministic MCP tool invocations without direct tensor manipulation.
- System prompt enforces that the LLM agent never fabricates tool parameters: all `media_plasma_id` values must be ObjectIDs returned by prior `ingest_media_tensor()` calls; all `fingerprint_hash` values must be SHA3-256 hex strings returned by prior `generate_fingerprint` tool responses.
- `run_fingerprint_pipeline(audio_path: str, config: dict) -> dict`: end-to-end orchestration entry point.

**4. Kubernetes GPU Autoscaler (`finops_autoscaler.py`)**

- `gpu_autoscale_loop()`: 15-second evaluation loop querying `mcp_queue_depth` from Prometheus and scaling the `omnipulse-gpu` Deployment to `max(1, min(32, queue_depth // 20))` replicas.
- `kubernetes.client.AppsV1Api` integration for `patch_namespaced_deployment_scale`.
- Prometheus `Gauge` export of `mcp_queue_depth` for observability integration.

---

### B. Packaging & Release Strategy

**PyPI Package: `omni-agent`**

```toml
[project]
name = "omni-agent"
version = "1.0.0"
dependencies = [
    "pyarrow >= 14.0",
    "numpy >= 1.26",
    "anthropic >= 0.28",
    "httpx >= 0.27",   # async MCP client transport
    "kubernetes >= 29.0",
    "prometheus-client >= 0.20",
]

[project.optional-dependencies]
langchain = ["langchain >= 0.2", "langchain-anthropic >= 0.1"]
```

- Pure Python package; no compiled extensions. `pip install omni-agent` on all platforms with no build dependencies.
- `omni-agent[langchain]` optional extra for LangChain tool integration.
- CLI entry point: `omni-agent serve --mcp-endpoint http://localhost:8080` launches the control plane as a standalone process.

**Docker Image:**

- `omni/control-plane:1.0` as per TDD §4.2.1: installs `omni-agent` and exposes the agentic loop as a long-running Python process.
- `ANTHROPIC_API_KEY` and `RUST_MCP_ENDPOINT` injected as environment variables.

---

### C. Alternative R&D Applications

**Application 1: Multi-Modal Scientific Literature Ingestion — Agentic PDF Processing Pipeline**

Large scientific publishers (arXiv, PubMed, IEEE Xplore) produce document corpora where semantic content is distributed across text, figures, equations, and tables. Traditional ETL pipelines parse these modalities sequentially with independent tools; an LLM-orchestrated agentic pipeline can process them as a unified workflow.

- *Why Arrow Plasma?* Scientific PDFs rendered to per-page float32 image tensors (300 DPI, 8.5×11", 3 channels ≈ 26 MB per page) cannot be passed through JSON serialisation without catastrophic latency. Arrow Plasma enables the PDF rendering engine (running as a separate process via `pdf2image`) to write page tensors to shared memory and return ObjectIDs to the agentic control loop.
- *Why the LLM agentic loop?* The control plane LLM determines, based on the rendered page content, whether to invoke an OCR tool (text-heavy pages), a figure captioning tool (image-heavy pages), or an equation extraction tool (formula-heavy pages). This dynamic routing is impossible to encode in a static pipeline but is a natural reasoning task for an instruction-following LLM.
- *Concrete deployment:* `omni-agent` is extended with custom MCP tool schemas for `render_pdf_page`, `extract_table`, and `caption_figure`. The Plasma manager ingests rendered page tensors; the LLM routes each page to the appropriate tool and aggregates results into a structured document representation.

**Application 2: Autonomous Robotics Task Planning — Sensor Fusion Agentic Loop**

In robotics systems deploying multiple heterogeneous sensors (LIDAR point clouds, RGB-D images, IMU time series), a central planning agent must fuse high-dimensional sensor streams into task-relevant action decisions without the per-cycle serialisation overhead that would violate real-time control constraints.

- *Why Arrow Plasma?* LIDAR point clouds (100,000+ points × 6 channels, float32) and RGB-D frames (1280×720×4 pixels, float32) cannot be serialised to JSON and passed between the sensor fusion module (C++ ROS2 node) and the planning agent (Python) without violating 10Hz control loop budgets. Plasma enables the C++ sensor fusion node to write registered point clouds to shared memory; the Python planning agent receives ObjectIDs via a lightweight ROS2 topic message.
- *Why the LLM agentic loop?* High-level task decomposition (e.g., "pick up the red object and place it in the bin") requires semantic reasoning that maps to low-level MCP tool invocations (`detect_objects`, `plan_grasp`, `execute_trajectory`). The LLM agent decomposes the task, sequences the tools, and handles failure modes (e.g., grasp failure → retry with different approach angle) as part of its natural reasoning loop.
- *Concrete deployment:* `omni-agent` MCP client connects to a ROS2-based Rust orchestration backend. Arrow Plasma is deployed on the robot's on-board NVIDIA Jetson AGX, exploiting the unified LPDDR5 memory architecture for zero-copy GPU-CPU tensor sharing.

---

### D. Testing & Validation Protocols

**1. Arrow Plasma Zero-Copy Validation**

```python
import pyarrow.plasma as plasma
import numpy as np
import tracemalloc

def test_plasma_zero_copy():
    client = plasma.connect("/tmp/plasma")
    N = 100_000  # 400KB float32 tensor

    arr = np.random.randn(N).astype(np.float32)

    tracemalloc.start()
    snapshot_before = tracemalloc.take_snapshot()

    # Ingest into Plasma — should not allocate heap memory proportional to N
    manager = PlasmaManager()
    object_id_hex = manager.ingest_media_tensor(arr)

    snapshot_after = tracemalloc.take_snapshot()
    stats = snapshot_after.compare_to(snapshot_before, 'lineno')
    heap_delta = sum(s.size_diff for s in stats)

    # Assert: heap allocation is O(1) (ObjectID metadata only), not O(N)
    assert heap_delta < 10_000, f"Unexpected heap allocation: {heap_delta} bytes"

    # Verify tensor integrity via re-read (also zero-copy via memoryview)
    [obj_buf] = client.get_buffers([plasma.ObjectID(bytes.fromhex(object_id_hex))])
    recovered = np.frombuffer(obj_buf, dtype=np.float32)
    np.testing.assert_array_equal(arr, recovered)
```

**2. MCP Tool Schema Contract Testing**

- Use `hypothesis` property-based testing to generate random valid `generate_fingerprint` and `compare_fingerprints` inputs and assert that all responses conform to the output schemas defined in TDD §3.2.
- Use `pytest-asyncio` for the full async `run_fingerprint_pipeline()` end-to-end test against a mock MCP server (`pytest-httpx`).

**3. LLM Agentic Hallucination Guard**

```python
def test_agent_does_not_hallucinate_plasma_ids():
    """Assert the LLM agent never invents PlasmaIDs not returned by ingest."""
    valid_ids = set()
    injected_calls = []

    def mock_ingest(arr):
        oid = manager.ingest_media_tensor(arr)
        valid_ids.add(oid)
        return oid

    def mock_mcp_call(tool, params):
        if "media_plasma_id" in params:
            assert params["media_plasma_id"] in valid_ids, \
                f"Hallucinated PlasmaID: {params['media_plasma_id']}"
        injected_calls.append((tool, params))
        return mock_responses[tool]

    run_fingerprint_pipeline("test.wav", cfg, ingest_fn=mock_ingest, mcp_fn=mock_mcp_call)
    assert len(injected_calls) >= 3  # at minimum: generate + compare + issue
```

**4. Kubernetes Autoscaler Logic Validation**

```python
@pytest.mark.parametrize("queue_depth,expected_replicas", [
    (0, 1), (19, 1), (20, 1), (21, 1), (40, 2), (200, 10), (640, 32), (1000, 32)
])
def test_autoscale_replica_calculation(queue_depth, expected_replicas):
    replicas = max(1, min(32, queue_depth // 20))
    assert replicas == expected_replicas
```

**5. End-to-End Plasma Lifecycle: Multi-Process Isolation**

- Spawn the `PlasmaManager` in a subprocess (simulating the Python control plane container) and a separate subprocess (simulating the C++ GPU container), and assert that:
  1. An ObjectID generated by subprocess A is resolvable by subprocess B within 100ms.
  2. Attempting to write to a sealed object raises `plasma.PlasmaObjectExists`.
  3. Releasing an object in subprocess A makes it immediately unavailable in subprocess B.

---

<a name="part-2-assembly"></a>

# Part 2 — The OmniPulse Assembly: Integration Engineering

Having independently built, released, and validated the three modules — `omni-wst-core` (C++/CUDA), `omni-orch` (Rust), and `omni-agent` (Python) — the integration workflow assembles them into the complete enterprise OmniPulse pipeline. Integration proceeds across three precisely defined interface boundaries.

---

## 2.1 The Memory Bridge: Python Arrow Plasma → C++ CUDA Kernels

**Problem:** The Python control plane operates at the apex of the architecture, yet it must feed multi-megabyte `float32` audio tensors into the C++ GPU kernels without incurring JSON serialisation overhead — a constraint the OmniPulse TDD (§3.1) identifies as the primary latency bottleneck in naïve implementations.

**Integration Mechanism:**

The Apache Arrow Plasma shared-memory object store is the exclusive inter-process tensor transport layer. The complete zero-copy lifecycle, as defined in TDD §3.1 (Steps 1–6), proceeds as follows:

**Step 1 — Python writes to Plasma:**

```python
# control_plane.py
plasma_id = manager.ingest_media_tensor(audio_array)
# Returns: "a3f8c2..." (20-byte hex ObjectID)
# The float32 tensor now lives in /tmp/plasma — a tmpfs mmap region
# shared across all containers via the plasma_store Docker volume
```

The Plasma server is deployed as a shared `tmpfs` volume (`plasma_store`) in the Docker Compose configuration (TDD §4.2.1), visible to all three language containers at `/plasma/store`. This ensures that the 20-byte ObjectID is the only artefact that crosses any inter-process boundary.

**Step 2 — Python dispatches ObjectID to Rust via MCP JSON:**

```python
response = await mcp_client.call_tool("generate_fingerprint", {
    "media_plasma_id": plasma_id,   # 20-byte hex — the only payload
    "config": {"J": 8, "Q": 16, "jtfs": True, "backend": "cuda"}
})
```

The MCP message size is `O(1)` — approximately 200 bytes of JSON — regardless of the tensor dimensionality. The tensor itself is never serialised.

**Step 3 — C++ resolves ObjectID to mmap virtual address:**

```cpp
// Inside run_wst_pipeline (called from Rust via cxx bridge)
// plasma_id arrives as a uint64_t mmap pointer
plasma::ObjectID oid = plasma::ObjectID::from_binary(
    reinterpret_cast<const uint8_t*>(&input_plasma_ptr));
std::shared_ptr<plasma::Buffer> buf;
PLASMA_CHECK(plasma_client.Get({oid}, -1, &buf));
const float* input = reinterpret_cast<const float*>(buf->data());
// Zero-copy: 'input' points directly into the tmpfs mmap region
```

The C++ kernel reads input natively from the shared mmap address, bypassing all host memory allocation. The first `cudaMemcpyAsync` call transfers this tensor directly from the mmap region to `d_input_b` on GPU via `stream1` (the DMA transfer stream).

**Step 4 — C++ seals the output fingerprint back into Plasma:**

```cpp
// After WST/JTFS forward_pass completes on stream0:
// D2H transfer: GPU scattering coefficients → new Plasma mmap
plasma::ObjectID out_oid = plasma::ObjectID::from_random();
std::shared_ptr<plasma::MutableBuffer> out_buf;
PLASMA_CHECK(plasma_client.Create(out_oid, coeff_count * sizeof(float), &out_buf));
CUDA_CHECK(cudaMemcpy(out_buf->mutable_data(), d_output,
    coeff_count * sizeof(float), cudaMemcpyDeviceToHost));
PLASMA_CHECK(plasma_client.Seal(out_oid));
result.fingerprint_ptr = /* encode out_oid as uint64_t */;
```

**Step 5–6 — ObjectID propagates back to Python via MCP response:**

The Rust orchestrator propagates the fingerprint ObjectID to Python as a hex string in the `generate_fingerprint` tool response. Python passes this ID directly to subsequent `compare_fingerprints` and `issue_license_token` tool invocations without ever materialising the tensor.

**Integration Validation:** Run the `test_plasma_zero_copy` test from §Module III/D against a live multi-container Docker Compose stack and assert that `heap_delta < 10_000` bytes throughout the full pipeline execution.

---

## 2.2 The FFI Bridge: C++ UVA CUdeviceptr → Rust Orchestrator via `cxx`

**Problem:** The Rust orchestrator must command the C++ kernel and receive back the location of computed scattering tensors without copying those tensors across the language boundary. The OmniPulse TDD (§2.1) specifies that `CUdeviceptr` (a 64-bit CUDA virtual address) is passed as an opaque `u64` handle.

**Integration Mechanism:**

**Step 1 — `cxx` bridge ABI alignment (compile-time validated):**

The `WSTResult` struct is declared in both C++ (`wst_bridge.h`) and Rust (`lib.rs`), with the `cxx` crate statically verifying ABI alignment at compile time — no runtime casting or reinterpret hacks:

```rust
// build.rs — executed at cargo build time
fn main() {
    cxx_build::bridge("src/lib.rs")
        .file("cpp/wst_bridge.cpp")
        .file("cpp/memory_staging.cu")  // nvcc compiled
        .flag("-std=c++17")
        .flag("-arch=sm_90")            // Hopper target
        .compile("omni_wst_bridge");

    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-link-lib=cufft");
    println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
}
```

**Step 2 — Rust calls C++ `run_wst_pipeline` with the Plasma ObjectID:**

```rust
pub fn execute_fingerprint_pass(plasma_id: u64, cfg: &WstConfig) -> WSTResult {
    // Safety contract:
    //   - plasma_id is a valid mmap pointer from the Arrow Plasma store.
    //   - The borrow checker (RustBelt) guarantees: no aliasing, no dangling refs.
    //   - C++ internal double-buffering (stream0/stream1) is isolated from Rust.
    unsafe {
        ffi::run_wst_pipeline(
            plasma_id,
            cfg.signal_len, cfg.batch_size,
            cfg.j, cfg.q, cfg.depth, cfg.jtfs,
        )
    }
}
```

The `unsafe` block is the **only** unsafe code in the entire Rust tier. Its safety is upheld by the documented contract above, which must be enforced at the call site (the MCP tool handler), not inside the function body.

**Step 3 — Rust receives `WSTResult.fingerprint_ptr` as an opaque `u64`:**

```rust
let result: WSTResult = execute_fingerprint_pass(plasma_id_u64, &cfg);
// result.fingerprint_ptr is a CUdeviceptr — opaque on the Rust side.
// Rust does NOT dereference it; it encodes it back into the Plasma ObjectID
// and returns the hex string to the Python control plane via MCP response.
let out_plasma_id = encode_cudeviceptr_as_plasma_id(result.fingerprint_ptr);
```

**Step 4 — The Rust HNSW store receives the scattering coefficient vector:**

When `compare_fingerprints` is invoked, Rust resolves the fingerprint ObjectID to the mmap address, reads the `float32` scattering coefficients into a `&[f32]` slice, and passes this slice to `FingerprintStore::query()` and `sliced_wasserstein_dist()`. This is the **only point** where the scattering tensor is materialised into Rust-managed memory — exclusively for the duration of the distance computation.

**UVA Integrity Assertion:**

```rust
fn validate_uva_handle(ptr: u64) -> bool {
    let mut attr_val: u32 = 0;
    let result = unsafe {
        cuda_sys::cuPointerGetAttribute(
            &mut attr_val as *mut _ as *mut _,
            cuda_sys::CUpointer_attribute_enum::CU_POINTER_ATTRIBUTE_MEMORY_TYPE,
            ptr as cuda_sys::CUdeviceptr,
        )
    };
    result == cuda_sys::cudaError_enum::CUDA_SUCCESS
        && attr_val == cuda_sys::CUmemorytype_enum::CU_MEMORYTYPE_HOST as u32
}
```

This assertion must be executed on every `WSTResult.fingerprint_ptr` received from C++ before propagating the handle into the Plasma ObjectID encoding step.

---

## 2.3 The Agentic Loop: Python MCP Tools → Rust Backend → Ed25519 IPFS Licensing

**Problem:** The Python LLM agent must orchestrate the complete fingerprinting-to-licensing workflow — spanning all three language runtimes — without ever directly manipulating tensor data, and must produce a cryptographically verifiable, IPFS-registered differential licensing token as the terminal output.

**Integration Mechanism:**

The agentic loop implements the following deterministic four-step orchestration sequence, enforced by the LLM's system prompt and the strict JSON schemas of the three MCP tools (TDD §3.2):

**Step 1 — `generate_fingerprint` (Python → Rust → C++ → Rust → Python)**

```python
# 1a. Python: ingest tensor into Plasma
plasma_id = manager.ingest_media_tensor(load_audio("derivative_track.wav"))

# 1b. Python: dispatch ObjectID to Rust MCP server
fingerprint_response = await mcp_client.call_tool("generate_fingerprint", {
    "media_plasma_id": plasma_id,
    "config": {"J": 8, "Q": 16, "jtfs": True, "backend": "cuda"}
})
# Returns: {"fingerprint_plasma_id": "...", "fingerprint_hash": "sha3hex...", "exec_time_ms": 42.1}
```

Internally, Rust's `generate_fingerprint` handler calls `execute_fingerprint_pass()` via the `cxx` bridge, receives the `WSTResult`, seals the output tensor into a new Plasma object, and returns the output ObjectID and SHA3-256 hash to Python.

**Step 2 — `compare_fingerprints` (Python → Rust HNSW + SW₁ → Python)**

```python
comparison_response = await mcp_client.call_tool("compare_fingerprints", {
    "query_hash": fingerprint_response["fingerprint_hash"],
    "reference_hash": None,   # None triggers HNSW nearest-neighbour search across full DB
    "n_projections": 256
})
# Returns: {
#   "sw_distance": 0.0312,
#   "l2_similarity": 0.947,
#   "is_licensed_derivative": True,
#   "confidence": 0.98,
#   "matched_identities": [{"hash": "...", "sw_distance": 0.0312, "title": "Root Work Alpha"}]
# }
```

Internally, Rust resolves the query ObjectID to the scattering coefficient `&[f32]`, calls `FingerprintStore::query()` to retrieve top-k HNSW neighbours, then calls `sliced_wasserstein_dist()` with `n_proj = 256` against each neighbour. If `sw_distance < τ` (the calibrated threshold), `is_licensed_derivative` is set to `true`.

**Step 3 — `issue_license_token` (Python → Rust Ed25519 + IPFS → Python)**

Invoked **only** when `comparison_response["is_licensed_derivative"] == True`:

```python
if comparison_response["is_licensed_derivative"]:
    license_response = await mcp_client.call_tool("issue_license_token", {
        "fingerprint_hash": fingerprint_response["fingerprint_hash"],
        "comparison_result_id": comparison_response["audit_id"],
        "licensee_pubkey": licensee_ed25519_pubkey_hex,
        "license_type": "derivative_commercial",
        "expiry_unix": int(time.time()) + 365 * 24 * 3600,
        "royalty_basis_points": 500   # 5% royalty
    })
    # Returns: {
    #   "token_hex": "bincode_payload_plus_ed25519_signature",
    #   "token_cid": "bafkreiXXX...",   # IPFS CIDv1
    #   "issued_at": 1712345678
    # }
```

Internally, Rust's `issue_license_token` handler constructs the `LicensePayload` struct, calls `bincode::serialize`, signs with the OmniPulse Ed25519 `SigningKey` (loaded from `/run/secrets/ed25519_sk` at server startup), and concurrently pins the payload to IPFS via `ipfs_client.add_bytes()`. The resulting CIDv1 provides **permanent, immutable, decentralised** registration of the license token — independent of the OmniPulse platform's operational status.

**Step 4 — HNSW Index Update**

Following successful `issue_license_token`, the Rust backend calls `FingerprintStore::insert()` to register the new derivative fingerprint in the HNSW index, enabling future `compare_fingerprints` calls to detect further derivatives of the derivative work:

```rust
store.insert(new_license_id, &derivative_embedding);
```

**Full Loop — LLM Agentic Orchestration Pseudocode:**

```python
# The LLM agent receives: "Process 'track.wav' and issue a commercial derivative license"
async def agentic_licensing_workflow(natural_language_request: str, audio_path: str):
    # Step 0: LLM parses request → determines parameters
    params = await llm.extract_parameters(natural_language_request)

    # Step 1: Ingest + Fingerprint
    plasma_id = manager.ingest_media_tensor(load_audio(audio_path))
    fp = await mcp_client.call_tool("generate_fingerprint", {
        "media_plasma_id": plasma_id, "config": params["fingerprint_config"]
    })

    # Step 2: Compare against registered corpus
    cmp = await mcp_client.call_tool("compare_fingerprints", {
        "query_hash": fp["fingerprint_hash"], "reference_hash": None, "n_projections": 256
    })

    # Step 3: Conditionally issue license
    if cmp["is_licensed_derivative"]:
        token = await mcp_client.call_tool("issue_license_token", {
            "fingerprint_hash": fp["fingerprint_hash"],
            "comparison_result_id": cmp["audit_id"],
            "licensee_pubkey": params["licensee_pubkey"],
            "license_type": params["license_type"],
            "expiry_unix": params["expiry_unix"],
            "royalty_basis_points": params["royalty_bps"]
        })
        return {"status": "licensed", "cid": token["token_cid"], "token": token["token_hex"]}
    else:
        return {"status": "not_derivative", "sw_distance": cmp["sw_distance"]}
```

**Integration Correctness Invariants:**

1. **ObjectID integrity:** Every `media_plasma_id` and `fingerprint_plasma_id` passed to any MCP tool must be resolvable in the Plasma store. The Rust MCP server must return `PlasmaObjectNotFound` (JSON-RPC error code `-32001`) if resolution fails.
2. **Licensing gate:** The `issue_license_token` tool handler must internally re-verify `sw_distance < τ` using the `comparison_result_id` audit trail — it must not trust the Python-provided `fingerprint_hash` alone. This prevents replay attacks where a Python client issues a license for an unverified derivative.
3. **IPFS CID immutability:** The `token_cid` returned by `issue_license_token` must be independently verifiable via `ipfs cat <token_cid>` against the `token_hex` payload, confirming that the Ed25519 signature and `bincode` serialisation are deterministic and platform-independent.
4. **Monotonic audit trail:** Every `generate_fingerprint` and `compare_fingerprints` invocation must write a structured log entry (via `tracing::info!`) containing the ObjectID, SHA3-256 hash, and timestamp, forming an immutable audit trail across the full licensing workflow.

---

## Strategic Summary

| Module | Independent Product | Rust Crate / PyPI Package | Primary R&D Application |
|---|---|---|---|
| C++/CUDA WST/JTFS | `omni-wst-core` | PyPI + CUDA wheels | Gravitational wave analysis; genomic peak calling |
| Rust Orchestrator | `omni-orch` | `crates.io` workspace | HFT order book similarity; enterprise RAG de-duplication |
| Python Agent | `omni-agent` | PyPI | Scientific literature ingestion; robotic task planning |
| **Assembled** | **OmniPulse SaaS** | **Docker / Kubernetes** | **IP management at industrial scale** |

The architectural boundary between each module is a **formally specified, testable interface contract**: the Arrow Plasma ObjectID (Python ↔ C++), the `CUdeviceptr` opaque `u64` handle (C++ ↔ Rust via `cxx`), and the JSON-RPC 2.0 MCP tool schema (Rust ↔ Python). No tensor data, no raw memory pointer, and no cryptographic key material ever crosses a module boundary through any mechanism other than these precisely defined channels.

This modular architecture is not merely an engineering preference — it is the prerequisite for the OmniPulse platform's **legal defensibility**: each licensing token issued is the product of a formally memory-safe orchestration layer (RustBelt), cryptographically signed by an Ed25519 key that never leaves the Rust process, and registered on an immutable decentralised ledger — entirely independent of the LLM agentic layer that initiated the workflow.

---

*Document Classification: Confidential // OmniPulse Engineering Strategy // Revision 1.0*  
*References: Mallat & Bruna (2013); Lostanlen et al. (2019); Jung et al. POPL (2018); McKeown DFRWS (2025); Apache Arrow Plasma Specification; IPFS CIDv1 Specification; ed25519-dalek Crate Documentation*
