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



 <br> <br> <br> <br> <br> <br> <br> <br>