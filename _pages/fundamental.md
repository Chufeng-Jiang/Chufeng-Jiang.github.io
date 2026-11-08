---

permalink: /infra/fundamental/
title: "Fundamentals"

author_profile: true

---

<!-- Enable LaTeX mathematical equations -->
<script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>

{% include_relative subnav.md %}

{% if site.google_scholar_stats_use_cdn %}
{% assign gsDataBaseUrl = "https://cdn.jsdelivr.net/gh/" | append: site.repository | append: "@" %}
{% else %}
{% assign gsDataBaseUrl = "https://raw.githubusercontent.com/" | append: site.repository | append: "/" %}
{% endif %}
{% assign url = gsDataBaseUrl | append: "google-scholar-stats/gs_data_shieldsio.json" %}

<span class='anchor' id='about-me'></span>



# Fundamental
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
- **K and V:** Always multiply by 2 to account for both tensors.
- **Precision:** FP16/BF16 = 2 bytes per element.
- **Memory Units:** 1 GB = $10^9$ bytes; 1 GiB = $2^{30}$ bytes.
- **GQA:** Grouped-Query Attention reduces KV Cache memory by using fewer KV heads than Query heads.
- **Memory Budget:** The formula excludes allocator overhead, fragmentation, and model weights.





 <br> <br> <br> <br> <br> <br> <br> <br>