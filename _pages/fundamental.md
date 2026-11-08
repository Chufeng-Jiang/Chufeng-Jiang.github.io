---

permalink: /infra/fundamental/
title: "Fundamentals"

author_profile: true

---
{% include_relative subnav.md %}

{% if site.google_scholar_stats_use_cdn %}
{% assign gsDataBaseUrl = "https://cdn.jsdelivr.net/gh/" | append: site.repository | append: "@" %}
{% else %}
{% assign gsDataBaseUrl = "https://raw.githubusercontent.com/" | append: site.repository | append: "/" %}
{% endif %}
{% assign url = gsDataBaseUrl | append: "google-scholar-stats/gs_data_shieldsio.json" %}

<span class='anchor' id='about-me'></span>



# Fundamental





## Model FLOPs Utilization (MFU)

### 1. Key Concepts

| Term   | Full Name                            | Definition                                                   |
| ------ | ------------------------------------ | ------------------------------------------------------------ |
| FLOP   | Floating-Point Operation             | A single floating-point operation, such as addition or multiplication. |
| FLOPS  | Floating-Point Operations Per Second | Measures computational throughput (computing speed).         |
| TFLOP  | Tera Floating-Point Operations       | \(10^{12}\) floating-point operations.                       |
| TFLOPS | Tera FLOPS                           | \(10^{12}\) floating-point operations per second.            |
| MFU    | Model FLOPs Utilization              | Measures how efficiently a model uses the theoretical peak computing capacity of the hardware. |

Key distinction:

- FLOP = Amount of computation (work).
- FLOPS = Computational speed (work per second).
- MFU = Fraction of theoretical peak compute utilized for model operations.

### 2. MFU Formula

\[ \boxed{\text{MFU}=\frac{\text{Model FLOPs per Step}/\text{Step Time}}{\text{Hardware Peak FLOPS}}\times100\%} \]

Equivalently:

\[ \text{MFU}=\frac{\text{Model Throughput}}{\text{Theoretical Peak Throughput}}\times100\% \]

### 3. Example Calculation

Given:

| Parameter                         | Value                              |
| --------------------------------- | ---------------------------------- |
| Model computation per step        | 700 TFLOP                          |
| Execution time per step           | 5 seconds                          |
| NVIDIA A100 BF16 peak performance | 312 TFLOPS (course-provided value) |

Step 1: Calculate model throughput.

\[ \frac{700\ \text{TFLOP}}{5\ \text{s}}=140\ \text{TFLOPS} \]

Step 2: Calculate MFU.

\[ \text{MFU}=\frac{140}{312}\times100\% \]



 <br> <br> <br> <br> <br> <br> <br> <br>