---

permalink: /infra/fundamental/
title: "Fundamentals"

author_profile: true

---

<!-- Enable LaTeX mathematical equations -->
<script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>

{% include_relative subnav.md %}

<style>
.page__content pre,
.page__content pre code,
.page__content .highlight pre,
.page__content .highlight pre code {
    font-size: 16px !important;
    line-height: 1.65 !important;
}
.page__content :not(pre) > code {
    font-size: 0.96em !important;
}
</style>



{% if site.google_scholar_stats_use_cdn %}
{% assign gsDataBaseUrl = "https://cdn.jsdelivr.net/gh/" | append: site.repository | append: "@" %}
{% else %}
{% assign gsDataBaseUrl = "https://raw.githubusercontent.com/" | append: site.repository | append: "/" %}
{% endif %}
{% assign url = gsDataBaseUrl | append: "google-scholar-stats/gs_data_shieldsio.json" %}

<span class='anchor' id='about-me'></span>



# Fundamental

Code and Markdown: [https://github.com/Chufeng-Jiang/AI_Infra/tree/main/fundamental](https://github.com/Chufeng-Jiang/AI_Infra/tree/main/fundamental)

{: .no_toc }

## Table of Contents
{: .no_toc }

* TOC
{:toc}

---
## Model FLOPs Utilization (MFU)

### 1. Key Concepts

| Term   | Full Name                            | Definition                                                   |
| ------ | ------------------------------------ | ------------------------------------------------------------ |
| FLOP   | Floating-Point Operation             | A single floating-point operation, such as addition or multiplication. |
| FLOPS  | Floating-Point Operations Per Second | Measures computational throughput (computing speed).         |
| TFLOP  | Tera Floating-Point Operations       | $10^{12}$ floating-point operations.                       |
| TFLOPS | Tera FLOPS                           | $10^{12}$ floating-point operations per second.            |
| MFU    | Model FLOPs Utilization              | Measures how efficiently a model uses the theoretical peak computing capacity of the hardware. |

Key distinction:

- FLOP = Amount of computation (work).
- FLOPS = Computational speed (work per second).
- MFU = Fraction of theoretical peak compute utilized for model operations.

### 2. MFU Formula

The Model FLOPs Utilization (MFU) is calculated as:

$$
\boxed{
\text{MFU} =
\frac{
\text{Model FLOPs per Step} / \text{Step Time}
}{
\text{Hardware Peak FLOPS}
}
\times 100\%
}
$$

Equivalently:

$$
\text{MFU} =
\frac{
\text{Model Throughput}
}{
\text{Theoretical Peak Throughput}
}
\times 100\%
$$


### 3. Example Calculation

**Given:**

| Parameter | Value |
| --- | --- |
| Model computation per step | 700 TFLOP |
| Execution time per step | 5 seconds |
| NVIDIA A100 BF16 peak performance | 312 TFLOPS |

**Step 1: Calculate model throughput.**

$$
\text{Model Throughput}
=
\frac{700\ \text{TFLOP}}{5\ \text{s}}
=
140\ \text{TFLOPS}
$$

**Step 2: Calculate MFU.**

$$
\text{MFU}
=
\frac{140}{312}
\times 100\%
\approx
\boxed{44.9\%}
$$

**Result:** The model achieves approximately 44.9%
of the GPU's theoretical BF16 peak performance.

---

## KV Cache Memory Calculation

### 1. Key Concepts

| Term | Definition |
| --- | --- |
| KV Cache | Stores previously computed Key (K) and Value (V) tensors during autoregressive generation. |
| Layers ($L$) | Number of Transformer layers. |
| KV Heads ($H_{kv}$) | Number of Key/Value heads per layer. |
| Head Dimension ($D$) | Number of elements in each head. |
| Bytes per Element ($b$) | Memory per numerical element, e.g., FP16/BF16 = 2 bytes. |
| Sequence Length ($S$) | Number of cached tokens. |
| Batch Size ($B$) | Number of sequences. |

### 2. KV Cache Memory Formula

**Memory per token:**

$$
\boxed{
M_{\text{token}}
=
2 \times L \times H_{kv} \times D \times b
}
$$

**Total KV Cache Memory:**

$$
\boxed{
\text{KV Cache Memory}
=
2 \times B \times S \times L \times H_{kv} \times D \times b
}
$$

The factor of **2** represents both Key (K) and Value (V).

### 3. Example: Llama-3-70B

**Given:**

| Parameter | Value |
| --- | --- |
| Transformer Layers | 80 |
| KV Heads | 8 |
| Head Dimension | 128 |
| Precision | FP16/BF16 (2 bytes) |
| Sequence Length | 8192 tokens |
| Batch Size | 1 |

**Step 1: Memory per layer per token**

$$
M_{\text{layer,token}}
=
2 \times 8 \times 128 \times 2
=
4096\ \text{bytes}
$$

**Step 2: Memory per token across all layers**

$$
M_{\text{token}}
=
4096 \times 80
=
327680\ \text{bytes}
$$

$$
\boxed{M_{\text{token}}=320\ \text{KiB}}
$$

**Step 3: Memory for an 8192-token sequence**

$$
M_{\text{KV}}
=
327680 \times 8192
=
2684354560\ \text{bytes}
$$

$$
\boxed{
M_{\text{KV}}
\approx 2.68\ \text{GB}
=2.5\ \text{GiB}
}
$$

### 4. Important Exam Notes

- **Linear Scaling:** KV Cache memory grows linearly with sequence length and batch size.
- **Memory Units:** 1 GB = $10^9$ bytes; 1 GiB = $2^{30}$ bytes.
- **GQA:** Grouped-Query Attention reduces KV Cache memory by using fewer KV heads than Query heads.
- **Memory Budget:** The formula excludes allocator overhead, fragmentation, and model weights.

---


## Performance Debugging and Scaling Efficiency

### 1. How Would You Debug a Slow Training Job?

**Answer:**

1. **Verify the workload and configuration**
   - Check model size, batch size, sequence length, and precision.
   - Verify the number of GPUs and distributed training configuration.
   - Make sure the input data and workload are consistent.

2. **Measure performance**
   - Record training step time and throughput (samples/s or tokens/s).
   - Measure GPU utilization, MFU, and memory usage.
   - Profile data loading, forward pass, backward pass, optimizer step, and communication.

3. **Identify potential bottlenecks**
   - **Compute-bound:** GPU arithmetic operations limit performance.
   - **Memory-bound:** GPU memory bandwidth or data movement limits performance.
   - **Communication-bound:** GPUs spend significant time waiting for collective communication.
   - **Data-loading-bound:** The CPU, storage, or preprocessing pipeline cannot feed data fast enough.
   - **Synchronization-bound:** Some GPUs wait for slower workers.

4. **Compare single-GPU and multi-GPU performance**
   - Run comparable workloads and configurations.
   - Measure scaling efficiency and communication overhead.

5. **Collect additional evidence if necessary**
   - Use PyTorch Profiler, NVIDIA Nsight Systems, or Nsight Compute.
   - Avoid claiming a root cause without profiling evidence.

**Key Takeaway:** Measure first, isolate the bottleneck, and optimize based on evidence.

---

### 2. What Is the Difference Between MFU and GPU Utilization?

**Answer:**

**MFU (Model FLOPs Utilization)** measures the useful model computation throughput relative to the GPU's theoretical peak computing capacity.

$$
\boxed{
\text{MFU} =
\frac{\text{Model FLOPs per Second}}
{\text{Theoretical Peak FLOPS}}
\times 100\%
}
$$

**GPU Utilization** commonly refers to the percentage of sampled time during which the GPU is executing at least one kernel, such as the utilization reported by `nvidia-smi`.

| Metric | MFU | GPU Utilization |
| --- | --- | --- |
| Measures | Useful model compute throughput | GPU active time |
| Unit | Percentage (%) | Percentage (%) |
| Main question | How efficiently is peak compute capacity used? | How often is the GPU busy? |
| High value means | High useful model compute throughput relative to peak | GPU frequently executes kernels |
| Limitation | Depends on the model FLOPs estimate and hardware peak | Does not directly measure computational efficiency |

**Example:**

A GPU can report 100% utilization but have relatively low MFU.

Possible reasons include:

- Memory bandwidth bottlenecks.
- Inefficient GPU kernels.
- Small matrix operations.
- Poor Tensor Core utilization.
- Communication or synchronization overhead.

**Key Takeaway:** High GPU utilization does not necessarily mean high computational efficiency.

---

### 3. How Would You Prove That an Optimization Is Effective?

**Answer:**

1. **Establish a baseline**
   - Record latency, throughput, GPU memory usage, and other relevant metrics before optimization.

2. **Maintain identical experimental conditions**
   - Same hardware and software environment.
   - Same model, dataset, batch size, and precision.
   - Same request patterns or training workload.

3. **Run repeated experiments**
   - Include warm-up iterations.
   - Repeat measurements to reduce noise.
   - Report median or mean and variability.

4. **Verify correctness**
   - Compare outputs against the baseline.
   - Check numerical tolerances.
   - Ensure model quality or training behavior is not unintentionally degraded.

5. **Measure additional costs**
   - GPU memory overhead.
   - CPU usage.
   - Communication overhead.
   - Compilation or initialization overhead.

**Speedup Formula:**

For a fixed workload:

$$
\boxed{
\text{Speedup} =
\frac{\text{Baseline Execution Time}}
{\text{Optimized Execution Time}}
}
$$

**Example:**

If the baseline takes 100 ms and the optimized version takes 60 ms:

$$
\text{Speedup}=\frac{100}{60}\approx1.67\times
$$

The execution time is reduced by 40%.

**Key Takeaway:** A valid optimization requires reproducible performance improvements, correctness verification, and a clear accounting of overhead.

---

### 4. Why Does Scaling From 8 GPUs to 64 GPUs Only Improve Throughput by 4x?

**Question:** A training system scales from 8 GPUs to 64 GPUs, but throughput increases by only 4x. What should you investigate?

**Answer:**

**Step 1: Calculate scaling efficiency.**

The GPU count increases by:

$$
\frac{64}{8}=8\times
$$

The observed throughput improvement is:

$$
\text{Speedup}=4\times
$$

Under ideal linear throughput scaling, the expected improvement would be 8x.

Therefore:

$$
\boxed{
\text{Scaling Efficiency}
=
\frac{\text{Observed Speedup}}
{\text{Ideal Speedup}}
\times100\%
}
$$

$$
\text{Scaling Efficiency}
=
\frac{4}{8}\times100\%
=
\boxed{50\%}
$$

**Step 2: Investigate potential bottlenecks.**

| Bottleneck | What to Investigate |
| --- | --- |
| Communication | All-Reduce, All-to-All, network bandwidth, and latency |
| Synchronization | GPU idle time and straggler workers |
| Workload imbalance | Uneven computational load across GPUs |
| Data loading | CPU preprocessing, storage bandwidth, and I/O |
| Batch size | Insufficient computation per GPU |
| GPU kernels | Kernel launch overhead and inefficient small operations |
| Memory | Memory bandwidth, allocation overhead, and fragmentation |

**Step 3: Break down execution time.**

Measure:

- Computation time.
- Communication time.
- Synchronization and waiting time.
- Data loading time.

Compare the 8-GPU and 64-GPU configurations using the same benchmark methodology.

**Step 4: Identify the next optimization.**

- If communication dominates, investigate communication overlap, collective algorithms, and interconnect performance.
- If GPUs are underutilized, investigate workload distribution and per-GPU batch size.
- If data loading dominates, optimize preprocessing, caching, and data pipeline parallelism.
- If some GPUs are slower, investigate load imbalance and synchronization.

**Important:** Before interpreting the 50% efficiency, verify that both measurements use comparable workloads and the intended scaling methodology. Strong scaling and weak scaling have different expectations.

**Key Takeaway:** GPU count and throughput numbers alone cannot identify the bottleneck. Profiling is required to determine whether the limitation comes from communication, computation, synchronization, or data loading.

## Memory Performance and Roofline Model

### 1. Capacity vs. Latency vs. Bandwidth

Three fundamental metrics describe memory system performance.

| Metric        | Definition                                 | Typical Units |
| ------------- | ------------------------------------------ | ------------- |
| **Capacity**  | Total amount of data that can be stored.   | KiB, MiB, GiB |
| **Latency**   | Time required to complete a memory access. | ns, μs        |
| **Bandwidth** | Amount of data transferred per second.     | GB/s, TB/s    |

**Key Distinction:**

- Capacity: **How much data can be stored?**
- Latency: **How long does one memory access take?**
- Bandwidth: **How much data can be transferred per second?**

A system can have high bandwidth but still experience high latency for individual memory requests.

**Example:**

- Reading one random memory location is often sensitive to latency.
- Scanning a large contiguous array is often sensitive to bandwidth.
- Storing a large model depends on memory capacity.

**Key Takeaway:** Large memory capacity does not imply low latency or high bandwidth.

---

### 2. How Are Latency and Bandwidth Connected?

Bandwidth and latency are different metrics, but concurrency connects them.

To sustain high bandwidth despite memory latency, a system must keep enough independent memory requests in flight.

**Bandwidth-Delay Product:**

$$
\boxed{
\text{Required In-Flight Data}
=
\text{Bandwidth}
\times
\text{Latency}
}
$$

This relationship is closely related to Little's Law.

**Example:**

Assume:

| Parameter               | Value     |
| ----------------------- | --------- |
| Target memory bandwidth | 89.6 GB/s |
| Memory latency          | 85 ns     |
| Cache line size         | 64 bytes  |

**Step 1: Calculate required in-flight data.**

$$
89.6 \times 10^9\ \text{B/s}
\times
85 \times 10^{-9}\ \text{s}
=
\boxed{7616} \text{    bytes}
$$

**Step 2: Convert to cache lines.**
$$
\text{Required Cache Lines}
=
\frac{7616}{64}
=
\boxed{119}
$$

Therefore, approximately **119 independent cache-line requests** must be in flight to sustain 89.6 GB/s under these assumptions.

**Important:** This is an idealized estimate assuming a stable transfer rate, the stated latency, and full 64-byte transfers. Real systems have additional restrictions.

**Key Takeaway:** High bandwidth requires sufficient memory-level parallelism to hide memory latency.

---

### 3. Why Does Memory-Level Parallelism Matter?

**Memory-Level Parallelism (MLP)** means processing multiple independent memory requests concurrently.

Consider two access patterns.

**Case A: Independent Memory Accesses**

Multiple requests can be issued without waiting for each other's results.

Examples:

- Reading independent array elements.
- Accessing unrelated memory addresses.
- Processing multiple independent data streams.

Higher concurrency can hide latency and improve achieved bandwidth.

**Case B: Dependent Memory Accesses**

Each request depends on the result of the previous request.

Example:

```c++
node = node->next;
node = node->next;
node = node->next;
```

The next memory address cannot be determined until the previous request finishes.

This creates a dependency chain, making latency difficult to hide.

| Property               | Independent Access                          | Dependent Access                       |
| ---------------------- | ------------------------------------------- | -------------------------------------- |
| Request concurrency    | Potentially high                            | Limited by dependencies                |
| Main limitation        | Often bandwidth                             | Often latency                          |
| Typical example        | Large array scan                            | Linked-list traversal                  |
| Potential optimization | Improve locality, batching, and parallelism | Reduce dependencies or reorganize data |

**Key Takeaway:** Independent requests can overlap; dependent requests often serialize.

---

### 4. Example: Estimating Single-Core Bandwidth

Assume one CPU core can maintain at most 16 independent outstanding cache-line reads.

| Parameter                    | Value     |
| ---------------------------- | --------- |
| Outstanding cache-line reads | 16        |
| Cache line size              | 64 bytes  |
| Memory latency               | 85 ns     |
| Target system bandwidth      | 89.6 GB/s |

**Step 1: Calculate in-flight data.**

$$
16 \times 64 = 1024\ \text{bytes}
$$

**Step 2: Estimate maximum sustained bandwidth.**

$$
\text{Bandwidth}
\approx
\frac{1024}{85 \times 10^{-9}}
$$

$$
\boxed{
\text{Bandwidth}
\approx 12.0\ \text{GB/s}
}
$$

**Step 3: Compare against target bandwidth.**

$$
\frac{89.6}{12.0}\approx7.5
$$

Under this simplified model, the core could sustain approximately 12 GB/s.

Achieving 89.6 GB/s would require about 7.5 times as much independent request capacity, potentially from additional cores or other sources of concurrency.

**Important:** This does NOT mean every CPU requires 8 threads to saturate memory bandwidth.

Actual performance depends on:

- Memory-level parallelism per core.
- Hardware prefetching.
- Cache hierarchy and cache hit rates.
- Memory controller behavior.
- Number of active cores.
- Request sizes and access patterns.

**Key Takeaway:** The number of outstanding memory requests can limit achieved bandwidth.

---

### 5. What Is Arithmetic Intensity?

**Arithmetic Intensity (AI)** measures how much computation is performed relative to the amount of data transferred.

$$
\boxed{
\text{Arithmetic Intensity}
=
\frac{\text{Floating-Point Operations}}
{\text{Bytes Transferred}}
}
$$

**Unit:** FLOP/Byte.

| Arithmetic Intensity | Interpretation                                        |
| -------------------- | ----------------------------------------------------- |
| Low AI               | Little computation per byte; more likely memory-bound |
| High AI              | More computation per byte; more likely compute-bound  |

Arithmetic intensity depends on the algorithm, implementation, data reuse, and level of the memory hierarchy being measured. For Roofline analysis, bytes transferred usually refer to a specified memory level, commonly DRAM.

---

#### 5.1. Example A: Vector Addition

Consider:

```c
c[i] = a[i] + b[i];
```

Assume each value uses FP32 (4 bytes).

For one output element:

| Operation                  | Amount              |
| -------------------------- | ------------------- |
| Read `a[i]`                | 4 bytes             |
| Read `b[i]`                | 4 bytes             |
| Write `c[i]`               | 4 bytes             |
| Floating-point computation | 1 addition = 1 FLOP |

**Total data transferred:**

$$
4+4+4=12\ \text{bytes}
$$

**Arithmetic Intensity:**
$$
AI=\frac{1}{12}
$$

$$
\boxed{AI\approx0.083\ \text{FLOP/Byte}}
$$

Vector addition has very low arithmetic intensity.

It is typically **memory-bandwidth-bound** when operating on large arrays that exceed the relevant cache capacity.

**Note:** This assumes ideal streaming reads and writes. Actual traffic may differ due to caching, write allocation, and other effects.

---

#### 5.2. Example B: Matrix Multiplication

Consider:

$$
C=A\times B
$$

where A, B, and C are FP32 matrices of size:

$$
1024 \times 1024
$$

**Step 1: Calculate FLOPs.**

For conventional dense matrix multiplication:

$$
\text{FLOPs}\approx2N^3
$$

Therefore:

$$
2\times1024^3
=
2,147,483,648\ \text{FLOPs}
$$

**Step 2: Calculate ideal data movement.**

Assume each input matrix is loaded once and the output matrix is written once. (2 + 1 = 3)

$$
\text{Bytes}
=
3\times1024^2\times4
=12,582,912  \text{   bytes}
$$

**Step 3: Calculate arithmetic intensity.**
$$
AI=
\frac{2\times1024^3}
{3\times1024^2\times4}
$$

$$
\boxed{
AI\approx170.7\ \text{FLOP/Byte}
}
$$

This is much higher than vector addition. Matrix multiplication offers substantial data reuse because the same input elements participate in many arithmetic operations.

**Important:** This is an idealized arithmetic intensity assuming effective data reuse. A poorly implemented matrix multiplication may transfer substantially more data.

**Key Takeaway:** Data reuse can dramatically increase arithmetic intensity.

---

### 6. What Is the Roofline Model?

The **Roofline Model** estimates an application's attainable floating-point performance based on two hardware limits:

1. Peak computational throughput.
2. Peak memory bandwidth.

The performance bound is:

$$
\boxed{
P_{\text{attainable}}
\leq
\min
\left(
P_{\text{peak}},
AI\times BW_{\text{peak}}
\right)
}
$$

Where:

- $P_{\text{attainable}}$: attainable performance (FLOPS).
- $P_{\text{peak}}$: theoretical peak computation throughput (FLOPS).
- $AI$: arithmetic intensity (FLOP/Byte).
- $BW_{\text{peak}}$: peak memory bandwidth (Bytes/s).

**Compute-Bound:**

Performance is primarily limited by computational throughput.

**Memory-Bound:**

Performance is primarily limited by the rate at which data can be transferred.

**Ridge Point / Machine Balance:**

The transition occurs at:

$$
\boxed{
AI_{\text{ridge}}
=
\frac{P_{\text{peak}}}{BW_{\text{peak}}}
}
$$

This value represents the arithmetic intensity required for the compute ceiling to become the limiting bound.

---

#### 6.1. Example: CPU Roofline Analysis

Assume a CPU has:

| Parameter                          | Value     |
| ---------------------------------- | --------- |
| CPU cores                          | 8         |
| Clock frequency                    | 4 GHz     |
| FP32 operations per cycle per core | 32        |
| Memory bandwidth                   | 89.6 GB/s |

**Step 1: Calculate theoretical peak FLOPS.**

$$
P_{\text{peak}}
=
8\times4\times32
$$

$$
\boxed{
P_{\text{peak}}=1024\ \text{GFLOPS}
}
$$

The 32 FLOPs/cycle/core assumes two 256-bit FP32 FMA units, each processing 8 values, with 2 FLOPs per FMA.

**Step 2: Calculate the ridge point.**

$$
AI_{\text{ridge}}
=
\frac{1024\ \text{GFLOPS}}
{89.6\ \text{GB/s}}
$$

$$
\boxed{
AI_{\text{ridge}}\approx11.4\ \text{FLOP/Byte}
}
$$

**Step 3: Compare algorithms.**

| Algorithm             | Arithmetic Intensity | Expected Bottleneck                  |
| --------------------- | -------------------- | ------------------------------------ |
| Vector Addition       | 0.083 FLOP/Byte      | Memory bandwidth                     |
| Matrix Multiplication | 170.7 FLOP/Byte      | Compute throughput, with ideal reuse |

These classifications use the simplified Roofline model.

**Key Takeaway:** When arithmetic intensity is below the ridge point, the memory-bandwidth ceiling is lower. Above the ridge point, the peak-compute ceiling is lower.

---

#### 6.2. Example: NVIDIA A100 GPU Roofline Analysis

Assume an NVIDIA A100 80GB configuration with the following theoretical specifications.

| Hardware Metric               | Value       |
| ----------------------------- | ----------- |
| Memory bandwidth              | 2039 GB/s   |
| FP32 peak compute             | 19.5 TFLOPS |
| BF16 Tensor Core peak compute | 312 TFLOPS  |

**Case A: FP32 Computation**

$$
AI_{\text{ridge,FP32}}
=
\frac{19.5\times1000}{2039}
$$

$$
\boxed{
AI_{\text{ridge,FP32}}\approx9.6\ \text{FLOP/Byte}
}
$$

**Case B: BF16 Tensor Core Computation**

$$
AI_{\text{ridge,BF16}}
=
\frac{312\times1000}{2039}
$$

$$
\boxed{
AI_{\text{ridge,BF16}}\approx153\ \text{FLOP/Byte}
}
$$

**Important:** The two ridge points differ because FP32 operations and BF16 Tensor Core operations have different theoretical peak throughput. The BF16 estimate assumes the workload uses the corresponding Tensor Core operations and peak-performance conditions. Always match the hardware peak with the numerical precision and operations used by the application.

---

### 7. How Can We Improve Memory-Bound Performance?

Common strategies include:

| Optimization          | Purpose                                                      |
| --------------------- | ------------------------------------------------------------ |
| Data Reuse            | Reduce repeated memory transfers                             |
| Tiling / Blocking     | Reuse data in cache or shared memory                         |
| Kernel Fusion         | Avoid unnecessary intermediate memory reads and writes       |
| Coalesced Access      | Improve GPU global-memory transaction efficiency             |
| Prefetching           | Overlap memory access with computation                       |
| Increased Concurrency | Hide memory latency through independent requests             |
| Reduced Precision     | Reduce bytes transferred per element when numerically acceptable |

**Example: Kernel Fusion**

Without fusion:

```c++
Kernel A
    ↓
Write intermediate result to global memory
    ↓
Kernel B
    ↓
Read intermediate result from global memory
```

With fusion:

```C++
Fused Kernel
    ↓
Reuse intermediate values in registers/shared memory
    ↓
Write final output
```

Kernel fusion can reduce global-memory traffic and kernel launch overhead. However, fusion may also increase register pressure and reduce occupancy, so performance should be measured rather than assumed.

**Key Takeaway:** Efficient GPU optimization often focuses on reducing data movement and increasing useful computation per transferred byte.

---

## C++ Fundamentals and Concurrency 

### 1. C++ and CUDA Kernels

When reading the implementation of a PyTorch or vLLM operator, begin with **inputs, outputs, computation, and ownership** instead of trying to understand every template parameter immediately.

**Example: vLLM-style CUDA activation kernel (excerpt)**

```c++
template <typename scalar_t, typename packed_t,
          scalar_t (*ACT_FN)(const scalar_t&, const float),
          packed_t (*PACKED_ACT_FN)(const packed_t&, const float),
          bool act_first, bool use_vec, bool HAS_CLAMP, bool use_256b = false>
__global__ void act_and_mul_kernel(
    scalar_t* __restrict__ out,          // output: [..., d]
    const scalar_t* __restrict__ input,  // input: [..., 2, d]
    const int d, const float limit, const float alpha, const float beta) {
    // ...
}
```

| Syntax                        | Meaning                                                      |
| ----------------------------- | ------------------------------------------------------------ |
| `scalar_t* out`               | Pointer to output elements (written by the kernel).          |
| `const scalar_t* input`       | Pointer to input elements; cannot modify them through this pointer. |
| `d`, `limit`, `alpha`, `beta` | Dimensions and numerical parameters.                         |
| `__global__`                  | CUDA kernel executed on GPU, launched from the host.         |
| `template<...>`               | Compile-time selection of types and implementation options.  |
| `__restrict__`                | No-aliasing access promise that may enable optimization; must be valid. |

**Small device function: SiLU**

```cpp
template <typename T>
__device__ __forceinline__ T silu_kernel(const T& x, const float alpha) {
    return (T)(((float)x) / (1.0f + expf((float)-x * alpha)));
}
```

Its mathematical operation is:

$$
\operatorname{SiLU}_{\alpha}(x)
= \frac{x}{1+e^{-\alpha x}}
= x\,\sigma(\alpha x)
$$

When $\alpha = 1$, it becomes the standard SiLU, $x\,\sigma(x)$.

- `__device__`: function executed on the GPU and callable from device code.
- `__forceinline__`: asks the compiler to aggressively inline the function.
- `(float)x`: converts to `float` for the intermediate computation.
- `(T)`: converts the result back to the template type.

**Takeaway:** First locate the data flow and mathematical operation; then examine types, layouts, vectorization, and kernel launch details. Understanding a small mathematical helper does not mean you have understood the entire GPU kernel.

---

### 2. Function Parameters: Value, Reference, and Pointer

**Example: C++ operator declarations**

```cpp
void silu_and_mul(torch::Tensor& out, torch::Tensor& input);

void rms_norm(torch::Tensor& out, torch::Tensor& input,
              std::optional<torch::Tensor> weight,
              double epsilon);
```

- `torch::Tensor& out`: reference to an existing Tensor object. The function may write to it, but examine the body to confirm behavior.
- `double epsilon`: scalar passed by value.
- `std::optional<torch::Tensor>`: the Tensor argument may be present or absent. An absent Tensor is not the same as a zero-element Tensor.

| Parameter    | How it is received     | Modification / nullability                                   |
| ------------ | ---------------------- | ------------------------------------------------------------ |
| `T v`        | By value               | Changes to the local parameter do not modify the caller's object itself. Copy behavior depends on `T`. |
| `T& r`       | By non-const reference | Can modify the referenced object; must refer to a valid object. |
| `const T& r` | By const reference     | Cannot modify the object through this reference in the usual way. |
| `T* p`       | By pointer             | Can be `nullptr`; may modify the pointed-to object.          |
| `const T* p` | Pointer to const       | Can be `nullptr`; cannot modify the pointed-to object through `p`. |

**Pass-by-value vs. reference**

```cpp
void change_copy(int value) { value = 9; }
void change_original(int& value) { value = 9; }

int number = 3;
change_copy(number);      // number is still 3
change_original(number);  // number becomes 9
```

**Important distinction: `std::vector` vs. `torch::Tensor`**

- Copying `std::vector<float>` ordinarily copies its elements, so the cost grows with the number of elements.
- Copying a `torch::Tensor` object usually copies a handle that shares the same underlying tensor data.
- Therefore, **passing an object by value does not always imply a deep copy of underlying storage**.
- A pointer gives an address but not necessarily its length, allocation owner, or validity.

**Takeaway:** Always distinguish the C++ object, its underlying data, and who owns that data.

---

### 3. `const`, Aliasing, and `__restrict__` 

Determine separately whether **the pointer may change** and whether **the pointed-to value may be modified through it**.

| Declaration            | Change `p` to another address? | Assign `*p` through `p`? |
| ---------------------- | ------------------------------ | ------------------------ |
| `const float* p`       | Yes                            | No                       |
| `float* const p`       | No                             | Yes                      |
| `const float* const p` | No                             | No                       |

**Aliasing:** Two different pointers may refer to the same underlying data.

**`__restrict__`:** A compiler optimization-related promise about aliasing. It is not a magic performance switch: if the actual access patterns violate the promise, the optimization is unsafe.

**Important limitations:**

- `const` restricts access through a particular interface. It does not prove that shared underlying memory can never change through another alias.
- A non-null pointer may still dangle if its resource has already been freed.
- A reference can also dangle if its target's lifetime has ended.

---

### 4. Object Lifetime, Ownership, and RAII

**RAII (Resource Acquisition Is Initialization)** ties resource cleanup to the lifetime of an owning object. Destructors release owned resources when the owner is destroyed.

```cpp
void compare_ownership() {
    float local_values[16];
    std::vector<float> managed_values(1 << 20);
    float* raw_values = new float[1 << 20];
    auto owned_values = std::make_unique<float[]>(1 << 20);
    delete[] raw_values;
}
```

| Resource                   | Who cleans it up?                                            |
| -------------------------- | ------------------------------------------------------------ |
| `float local_values[16]`   | Local array lifetime ends with the scope.                    |
| `std::vector<float>`       | Vector destructor releases its elements' storage.            |
| `new float[...]`           | Must be matched with `delete[]` on the appropriate ownership path. |
| `std::unique_ptr<float[]>` | Automatically frees its owned array.                         |
| `std::shared_ptr<T>`       | Shared ownership; resource is normally released when the last owner goes away. |

**Memory calculation from the source:**

$$
2^{20} = 1{,}048{,}576\ \text{elements}
$$

$$
1{,}048{,}576 \times 4\ \text{bytes}
= 4{,}194{,}304\ \text{bytes}
= \boxed{4\ \text{MiB}}
$$

**Common bugs:**

- **Memory leak:** allocated resources are not released.
- **Dangling pointer/reference:** a resource is accessed after its lifetime ends.
- **Use-after-free:** a particular case of invalid access after deallocation.

**Tensor-specific note:** `tensor.data_ptr()` provides access to a data address, but that raw pointer does not independently extend the Tensor storage lifetime. Check whether the data is on CPU or GPU and keep the owner valid.

**Takeaway:** `unique_ptr` and `shared_ptr` differ in ownership policy, not the amount of memory required by the array they manage.

---

### 5. C++ Templates, `if constexpr`, and Type Dispatch

A **template** is a compile-time mechanism for generating implementations for selected types or settings. When reading a complicated templated CUDA kernel, first imagine a concrete type such as `float`, then investigate the other specializations.

**Compile-time branch:**

```cpp
if constexpr (HAS_CLAMP) {
    gate = (scalar_t)fminf((float)gate, limit);
}
```

- `HAS_CLAMP` is a compile-time template parameter in this example.
- If `HAS_CLAMP` is `true`, the compiled specialization includes the clamp branch.
- If `false`, this branch is discarded for that specialization.
- A normal runtime `if` evaluates a condition at execution time, although the compiler may still optimize it away if the condition is provably constant.

**Type dispatch:** An operator may inspect a runtime Tensor's `dtype`, then select an already compiled specialization via a `DISPATCH` macro or equivalent mechanism.

**Cautions:**

- More template combinations can increase compilation time and binary size.
- `reinterpret_cast` changes how data is interpreted; it **does not numerically convert** values into another dtype. Correct alignment and access rules still matter.
- `torch::`, `at::`, and `c10::` denote namespace or scope names; `namespace py = pybind11;` is a namespace alias.

---

### 6. Compilation, Linking, Loading, and ABI

**Conceptual build pipeline:**

```c++
Headers (.h) + implementation (.cpp / .cu)
                  |
             Compilation
           g++ / nvcc -> .o
                  |
                Linking
               _C.so
                  |
          Runtime loading
          Python import
```

| Stage                       | Purpose                                                      | Example problem                                              |
| --------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Preprocessing / compilation | Find headers; check syntax and types; compile source files into object files. | `file not found`, type mismatch                              |
| Linking                     | Resolve references between object files and libraries; produce a shared library. | Undefined symbol / missing implementation                    |
| Runtime loading             | Load `.so` and its dependencies into the Python process.     | Missing library, incompatible ABI or Python/framework version |

**CMake example from the lesson:**

```c++
add_library(_C SHARED ${VLLM_EXT_SRC})
target_include_directories(_C PRIVATE csrc)
target_link_libraries(_C PRIVATE ${TORCH_LIBRARIES} CUDA::cudart)
```

- `add_library`: source files and shared-library build target.
- `target_include_directories`: search locations for included headers.
- `target_link_libraries`: libraries required for linking.
- `c++filt` can help demangle C++ symbol names when investigating an `undefined symbol` error.
- **ABI (Application Binary Interface)** describes binary-level calling and data conventions. Build-time and runtime Python, framework, and compiler compatibility may matter.
- Template definitions normally must be visible at the instantiation point, but explicit instantiation is another valid organization strategy.

**Takeaway:** Locate whether the failure occurs during compilation, linking, or import before changing build settings.

---

### 7. Python–C++ Integration: pybind11 and PyTorch Operators

**A. pybind11: expose a C++ function to Python**

```cpp
#include <pybind11/pybind11.h>
namespace py = pybind11;

int add(int i, int j) { return i + j; }

PYBIND11_MODULE(example, m) {
    m.doc() = "pybind11 example plugin";
    m.def("add", &add, "A function that adds two numbers");
}
```

Once the compiled module has been installed correctly:

```python
import example
print(example.add(1, 2))  # 3
```

- `PYBIND11_MODULE(example, m)` defines the module's Python import name.
- `m.def("add", &add, ...)` connects the exported Python name to the C++ function.



![image-20261108213923232](F:\GitHub\Chufeng-Jiang.github.io\_pages\fundamental.assets\image-20261108213923232.png)

**B. PyTorch operator registration (course excerpt)**

```cpp
TORCH_LIBRARY_FRAGMENT(CONCAT(TORCH_EXTENSION_NAME, _custom_ar), custom_ar) {
    custom_ar.def(
        "qr_all_reduce(int fa, Tensor inp, Tensor out, int quant_level, bool "
        "cast_bf2half) -> ()");
    custom_ar.impl("qr_all_reduce", torch::kCUDA, &qr_all_reduce);
}
```

- `def`: declares the operator schema (name, arguments, return type).
- `impl`: associates the operator with a device-specific implementation (CUDA here).
- Registration does not, by itself, provide everything needed for autograd, shape inference, or compilation into larger graphs.

**Conceptual Python-to-GPU execution path** (actual paths differ by operator/version):

```c++
Python API
    -> Binding / registered operator
    -> PyTorch Dispatcher
    -> C++ implementation (shapes, dtype dispatch, launch config)
    -> CUDA kernel running on GPU SMs
```

![image-20261108220331314](F:\GitHub\Chufeng-Jiang.github.io\_pages\fundamental.assets\image-20261108220331314.png)

**Takeaway:** To follow a Python operator into native code, identify the current layer (API, binding, dispatcher, C++ implementation, GPU kernel) and what it is responsible for.

---

### 8. Multithreading: Data Race vs. False Sharing

Multiple threads can access shared process memory. Correctness comes before performance.

| Concept       | Definition                                                   | Consequence                                                  |
| ------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| `std::thread` | Creates a C++ thread of execution.                           | Work can run concurrently.                                   |
| `std::mutex`  | Synchronizes access to protected shared data.                | Helps prevent conflicting unsynchronized access.             |
| Data race     | Conflicting unsynchronized accesses to the same memory location, with at least one write. | Undefined behavior in C++.                                   |
| False sharing | Different threads repeatedly write different locations on the same cache line. | Cache-coherence traffic and potential performance loss, even without a data race. |
| Reduction     | Combine partial results into a final result.                 | Often helps reduce contention and shared writes.             |

**Example: local accumulation, then one shared write**

```cpp
std::vector<double> partial(nthreads, 0.0);
auto worker = [&](int t) {
    double local = 0;
    for (size_t i = t; i < n; i += nthreads)
        local += data[i];
    partial[t] = local;
};
```

`[&]` means the lambda captures used outer variables by reference. The referenced objects must remain valid throughout the worker's use.

**Two-thread example using data = [1, 2, 3, 4, 5, 6]:**

- Thread 0 visits indices `0, 2, 4`: $1+3+5=9$.
- Thread 1 visits indices `1, 3, 5`: $2+4+6=12$.
- Final reduction: $9+12=\boxed{21}$.

This loop writes each `partial[t]` only once, reducing repeated shared-memory writes compared with `partial[t] += data[i]` inside the loop. The source excerpt omits creating/joining threads and the final reduction; these steps would be required in a complete implementation.

**Takeaway:** Different threads writing different elements need not cause a data race, but repeatedly writing elements on one cache line may still cause false sharing. Always verify correctness before evaluating speedup.

---

### 9. Common Exam Pitfalls

1. `GPU-Util` and useful compute throughput are not the same thing; likewise, `__global__` tells you where code executes, not whether it is efficient.
2. `T&` permits modification but does not prove the function modifies the caller's object.
3. `T v` means an object is passed by value, but the underlying storage may still be shared (e.g., Tensor handle).
4. `const T&` is not a guarantee that all underlying shared data is immutable.
5. A non-null pointer can still be invalid after the pointee's lifetime ends.
6. `std::unique_ptr` automatically manages owned memory; `new[]` should be paired with the correct `delete[]` ownership path.
7. `if constexpr` is a compile-time branch; dtype dispatch can select compiled implementations at runtime.
8. `reinterpret_cast` is not numerical dtype conversion.
9. A successful compile does not guarantee a successful link or `import`.
10. Registering a CUDA operator is not automatically enough to support autograd or shape inference.
11. Data races are correctness errors; false sharing is primarily a performance issue.
12. More CPU threads do not guarantee linear speedup. <br>
