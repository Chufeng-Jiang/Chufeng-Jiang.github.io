---

permalink: /infra/Pytorch_and_DeepLearning/
title: "Pytorch and DL"

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



# Pytorch and DL
{: .no_toc }

## Table of Contents
{: .no_toc }

* TOC
{:toc}
## 1. Tensors and Memory Layout: Shape, Stride, View vs. Copy

### 1.1. A tensor consists of storage and metadata

A dense strided tensor does not have to own a newly allocated, contiguous array. It references **storage**, together with metadata specifying how to interpret it:

| Property               | Meaning                                                  | Example                     |
| ---------------------- | -------------------------------------------------------- | --------------------------- |
| `shape` / `sizes()`    | Number of logical elements along each axis               | `(3,4)`                     |
| `stride` / `strides()` | Storage elements skipped when one index increases by one | `(4,1)`                     |
| `storage_offset`       | Element offset from the beginning of shared storage      | `0` or `1`                  |
| `dtype`                | Representation and bytes per element                     | FP32 = 4 B                  |
| `device`               | CPU or accelerator location                              | CPU, CUDA GPU               |
| storage                | Actual backing memory shared by compatible views         | 12 values in one allocation |

For a two-dimensional tensor, the element storage index is:

$$
\boxed{\mathrm{index}(i,j)=\mathrm{offset}+i\cdot\mathrm{stride}[0]+j\cdot\mathrm{stride}[1]}
$$

For a rank-$n$ tensor:

$$
\boxed{\mathrm{index}(i_0,\ldots,i_{n-1})=\mathrm{offset}+\sum_{k=0}^{n-1}i_k\,s_k}
$$

**Example.** A tensor containing `0..11` has shape `(3,4)` and strides `(4,1)`. The element at `[1,2]` is found at `0+1×4+2×1=6`. After transposition, the shape becomes `(4,3)` and strides become `(1,4)` **without moving the 12 values**. A slice selecting column 1 has shape `(3,)`, stride `(4,)`, offset `1`, and values `[1,5,9]`.

```cpp
#include <torch/torch.h>
#include <ATen/TensorIndexing.h>
#include <iostream>

int main() {
    auto t = torch::arange(12).reshape({3, 4});
    auto transposed = t.transpose(0, 1);   // View: no element copy
    auto col = t.index({at::indexing::Slice(), 1});
    std::cout << "t[1,2]=" << t.index({1, 2}).item<int64_t>() << "\n";
    std::cout << "transposed shape=" << transposed.sizes() << "\n";
    std::cout << "column=" << col << "\n";  // [1, 5, 9]
}
```

**PyTorch vs. NumPy:** PyTorch stride counts **elements**; NumPy stride counts **bytes**. A `(3,4)` FP32 tensor has PyTorch strides `(4,1)`, but an equivalent NumPy array has byte strides `(16,4)`.

**Memory calculation:** `torch.arange(12)` uses int64 by default in the book's example: `12 × 8 = 96 bytes`. The original tensor, transpose, and basic slice can all refer to that same allocation. The tensor object's metadata and other bookkeeping still consume additional memory.

### 1.2. Contiguity describes access order—not whether memory is fragmented

Under ordinary row-major layout, each successive logical element follows the expected storage order. For a nonempty, conventional dense tensor with non-singleton axes:

$$
\mathrm{stride}[-1]=1,\qquad
\mathrm{stride}[k]=\mathrm{shape}[k+1]\,\mathrm{stride}[k+1].
$$

Singleton and empty dimensions require special handling, and PyTorch recognizes multiple memory formats. **A slice can share storage with its parent yet be contiguous itself.**

- `is_contiguous()` checks a specified memory format; it does not diagnose allocator fragmentation.
- `contiguous()` copies *only if necessary* for the requested layout. It **does not guarantee** an independent copy.
- `clone()` creates independent data storage (although its result can remain connected in the autograd graph).
- `transpose()` / `permute()` generally change metadata and share storage; the next kernel may nevertheless run slower because access strides changed.

### 1.3. `view()` versus `reshape()`

| Operation               | Shares storage?        | Could copy?             | Key rule                                           |
| ----------------------- | ---------------------- | ----------------------- | -------------------------------------------------- |
| `view(new_shape)`       | Yes, if successful     | No                      | New shape must be representable with valid strides |
| `reshape(new_shape)`    | Maybe                  | Yes                     | Reuses storage where possible, otherwise copies    |
| `transpose` / `permute` | Usually yes            | Usually no              | Reorders dimension interpretation                  |
| `clone()`               | No                     | Yes                     | Explicit independent allocation                    |
| `contiguous()`          | Maybe                  | Yes                     | Copies if needed for target format                 |
| `expand()`              | Yes                    | No input-element copies | Expanded axes can have stride 0                    |
| `repeat()`              | No (new repeated data) | Yes                     | Materializes repeated values                       |

**Noncontiguous does not always imply `view()` failure.** With `x` of shape `(2,3,4)` and stride `(12,4,1)`, set `y=x.transpose(0,1)`. Then `y` has shape `(3,2,4)` and stride `(4,12,1)`:

- `y.view({3,2,2,2})` can work with strides `(4,12,2,1)` because the last axis is simply split.
- `y.view({6,4})` cannot merge the first two dimensions using one fixed stride; it fails.
- `y.reshape({6,4})` can succeed by copying, but you should not assume every `reshape()` performs a copy.

```cpp
#include <torch/torch.h>
#include <iostream>
int main() {
    auto x = torch::arange(24).reshape({2, 3, 4});
    auto y = x.transpose(0, 1);                 // shape {3,2,4}
    auto split = y.view({3, 2, 2, 2});          // Valid view
    auto flattened = y.reshape({6, 4});        // May allocate a copy
    auto guaranteed_copy = y.clone();          // Independent storage
    std::cout << split.sizes() << " " << flattened.sizes() << "\n";
}
```

### 1.4. `channels_last` changes layout without changing logical `NCHW` shape

For input `(N,C,H,W)=(10,3,32,32)`:

| Memory format           | Shape          | Illustrative strides |
| ----------------------- | -------------- | -------------------- |
| Default contiguous NCHW | `(10,3,32,32)` | `(3072,1024,32,1)`   |
| `channels_last`         | `(10,3,32,32)` | `(3072,1,96,3)`      |

The logical axes are still NCHW, but channels of one pixel are adjacent in channels-last storage. Certain convolution implementations can exploit this layout. **Conversion costs and downstream operator support must be included in the benchmark.** The improvement cited in the PDF is from a particular cuDNN/ResNet mixed-precision experiment, not a guaranteed speedup.

### 1.5. The performance lesson: a metadata-only operation can make later work expensive

Suppose a transposed FP32 view has stride `4096` along the axis being traversed. Consecutive logical reads may be **16 KiB apart** (`4096×4`). Under the PDF's simplified 64-byte cache-line illustration, using only one 4-byte value per line yields `4/64=1/16` useful payload. This does not prove a real machine's cache-line size or fully explain any benchmark.

**Decision:** Compare `noncontiguous_op` with `copy_to_contiguous + contiguous_op` over the **same entire workload**, particularly when the contiguous copy can be reused across many operations.

**Exam traps:** Printed tensor values do not reveal aliasing; `reshape()` does not promise copying; `contiguous()` does not promise a fresh allocation; noncontiguous does not automatically mean slow.

---

## 2. Broadcasting: Shape Semantics, Zero Strides, and Hidden Memory Growth

### 2.1. Broadcasting rules

1. Align shapes from the **right**; conceptually pad missing leading dimensions with size `1`.
2. Two axes are compatible if their sizes match or one is `1`.
3. Expand singleton axes to the required logical size; a view can use **stride 0** so the underlying input element is reused.
4. Check the *meaning* of each axis. A legal broadcast can still implement the wrong loss function.

**Worked shape:** `(8,1,6,1)` and `(7,1,5)` become `(8,1,6,1)` and `(1,7,1,5)`, producing **`(8,7,6,5)`**. However, `(4,3)` and `(4,)` are incompatible: their rightmost sizes are `3` and `4`. To add one scalar per row, use `(4,1)`; to add the same three feature offsets to every row, use `(3,)`.

### 2.2. The silently wrong regression loss

Suppose `pred` has shape `(B,)`, while `target` has shape `(B,1)`.

$$
(B,)\;\text{broadcast with}\;(B,1)\quad\longrightarrow\quad(B,B).
$$

For `B=1024`, this creates **1,048,576 pairwise errors** instead of 1,024 aligned errors. One FP32 error tensor grows from `4 KiB` to `4 MiB` (not counting further intermediates). For values `[0,1,2,3]`, aligned mean squared error is `0`; unintended full pairwise mean squared error is **`2.5`**.

```cpp
#include <torch/torch.h>
#include <iostream>
int main() {
    auto pred = torch::arange(4, torch::kFloat32); // {4}
    auto target = pred.reshape({4, 1});            // {4,1}
    auto wrong = (pred - target).square().mean();  // {4,4} then mean
    auto correct = (pred - target.squeeze(1)).square().mean();
    std::cout << "wrong=" << wrong.item<float>()
              << ", correct=" << correct.item<float>() << "\n";
}
```

**Safe habit:** Assert the task-required shape **before loss calculation**. Do not indiscriminately remove every singleton axis; batch size 1 and intentional broadcasting are important exceptions.

### 2.3. `expand()` vs. `repeat()` and in-place writes

A 4-element FP32 bias occupies **16 bytes**. Expanded to `(1000,4)`, it can retain that same 16-byte storage with strides `(0,1)`. Repeating it into 1,000 independent rows materializes **16,000 bytes**. The *result of arithmetic on an expanded input*, however, ordinarily has its own storage.

Multiple logical elements of a stride-0 view map to the same physical address. Writing different values into them is ambiguous. PyTorch does not make every expanded view read-only, so avoid unsafe in-place operations on overlapping elements. An in-place operation cannot expand the *destination's* shape.

### 2.4. `keepdim` preserves the intended broadcast axis

For `x.shape=(32,768)`, per-row centering requires means of shape `(32,1)`:

$$
\mu_i=\frac{1}{768}\sum_{j=1}^{768}x_{ij},\qquad y_{ij}=x_{ij}-\mu_i.
$$

Without keeping the feature axis, a mean of shape `(32,)` is incompatible with `(32,768)`, and an accidentally square matrix can make the incorrect operation *silently legal*.

```cpp
#include <torch/torch.h>
int main() {
    auto x = torch::randn({32, 768});
    auto row_mean = x.mean(/*dim=*/{1}, /*keepdim=*/true); // {32,1}
    auto centered = x - row_mean;                         // {32,768}
    TORCH_CHECK(centered.sizes() == x.sizes(), "Unexpected result shape");
}
```

### 2.5. Broadcasted gradients are summed across replicated axes

For `y=x+b`, with `x:(N,M)` and `b:(1,M)`, the bias is used in every row. Consequently:

$$
\boxed{\frac{\partial L}{\partial b_j}=\sum_{i=1}^{N}\frac{\partial L}{\partial y_{ij}}}
$$

If the upstream gradient is `[[1,2],[3,4]]`, then `b.grad=[[4,6]]`. **The backward sum is not an additional average.** If the loss used `.mean()`, that normalization is already represented in its upstream gradient. PyTorch may implement this reduction with `sum_to_size`-style logic.

### 2.6. A broadcast that creates a 4 GiB intermediate

With `X:(1024,1024)`, the expression `X[:,None,:] - X[None,:,:]` has shape `(1024,1024,1024)`.

$$
1024^3 \times 4\;\mathrm{B}=4\;\mathrm{GiB}.
$$

The input is only `4 MiB`, and the desired output of **pairwise squared Euclidean distances** is just `(1024,1024)` or `4 MiB`. Avoid materializing all feature differences by using:

$$
\boxed{D_{ij}=\|x_i\|^2+\|x_j\|^2-2x_i^\top x_j}
$$

```cpp
#include <torch/torch.h>
int main() {
    auto X = torch::randn({1024, 1024}, torch::kFloat32);
    auto sq_norm = X.square().sum(/*dim=*/1, /*keepdim=*/true); // {1024,1}
    auto D = sq_norm + sq_norm.transpose(0, 1) - 2 * X.matmul(X.t());
    D = D.clamp_min(0); // Compensate for small negative roundoff errors
}
```

The matrix formulation still has `(N,N)` outputs and temporaries. It computes **squared distances**; `torch.cdist` computes Euclidean distances unless otherwise specified. This example also motivates avoiding the full `(B,H,S,S)` attention matrix when using memory-efficient attention implementations.

**Exam traps:** An expanded input can be cheap while the output is huge; a shape-compatible operation can be semantically wrong; `expand` backward **sums** contributions.

---

## 3. Autograd and Backpropagation

*Source: PDF pp. 26–37.*

### 3.1. Chain rule and multiple gradient paths

For $z=xy+x$, with $x=2$, $y=3$:

$$
z=8,\qquad\boxed{\frac{\partial z}{\partial x}=y+1=4},\qquad\frac{\partial z}{\partial y}=x=2.
$$

The two uses of `x` contribute separately: multiply-path gradient `3`, add-path gradient `1`. They **must be added**. Replacing `+=` with `=` loses a contribution.

For vectors $a=[2,3]$, $b=[4,5]$, $c=[1,1]$ and $L=\sum(a\odot b+c)$:

| Quantity | Value    |
| -------- | -------- |
| `a*b`    | `[8,15]` |
| `a*b+c`  | `[9,16]` |
| `L`      | `25`     |
| `dL/da`  | `[4,5]`  |
| `dL/db`  | `[2,3]`  |
| `dL/dc`  | `[1,1]`  |

```cpp
#include <torch/torch.h>
#include <iostream>
int main() {
    auto opt = torch::TensorOptions().dtype(torch::kFloat32).requires_grad(true);
    auto a = torch::tensor({2.0f, 3.0f}, opt);
    auto b = torch::tensor({4.0f, 5.0f}, opt);
    auto c = torch::tensor({1.0f, 1.0f}, opt);
    auto loss = (a * b + c).sum();
    loss.backward();
    std::cout << "loss=" << loss.item<float>() << "\n";
    std::cout << "a.grad=" << a.grad() << "\n";
    std::cout << "b.grad=" << b.grad() << "\n";
}
```

### 3.2. Dynamic computation graph and saved tensors

- `requires_grad` requests derivative tracking for relevant operations.
- `grad_fn` identifies the **backward operation**, not the gradient value. In ordinary Python PyTorch, user-created trainable **leaf** tensors have no `grad_fn`, while operation results do.
- `.grad` on leaves accumulates gradient results. Intermediate tensors generally do not retain their `.grad` unless requested with `retain_grad()`.
- Backward nodes are linked by dependencies (`next_functions` in the Python representation). The engine propagates upstream gradients according to the chain rule and combines path contributions.
- Multiplication backward needs the other operand's forward value; addition need not save both numeric inputs; softmax backward may need its output. **Saved tensors can prolong memory lifetime without duplicating every underlying buffer.**

### 3.3. Reverse-mode autodiff versus finite differences

Numerical central difference for one parameter:

$$
\frac{\partial L}{\partial \theta_i}\approx
\frac{L(\theta_i+\varepsilon)-L(\theta_i-\varepsilon)}{2\varepsilon}.
$$

Testing all $N$ parameters this way needs approximately **`2N` forward evaluations**. Reverse-mode autodiff obtains the full gradient of a **scalar loss** via one backward traversal of the computation graph, making it advantageous for many trainable parameters and few scalar outputs.

For dense Transformer training, a common course-level approximation is:

$$
\boxed{F_{\mathrm{forward}}\approx2ND,\qquad F_{\mathrm{backward}}\approx4ND,\qquad F_{\mathrm{train}}\approx6ND}
$$

where $N$ = number of parameters and $D$ = token count processed. These are **rough FLOP models**, not exact counts for every architecture. The PDF's GPT-2-small example uses $N=124\,\mathrm{M}$, $D=8192$: forward $\approx2\,\mathrm{TFLOP}$; with an assumed A100 usable rate of `124.8 TFLOPS`, an idealized forward is around `16 ms`. Finite differences across all parameters would then require roughly **46 days** under those same simplistic assumptions, versus a single backward on the order of tens of milliseconds.

Reverse-mode calculations can be understood as **vector–Jacobian products (VJPs)**: the upstream gradient is combined with local Jacobian information *without building the enormous complete Jacobian matrix*.

### 3.4. Topological ordering versus gradient accumulation

For `u=x*y; L=u+u`, the backward gradient of `u` is `2`, but the multiplication's backward node should execute once after its contributions are combined. A micrograd implementation therefore performs a dependency-respecting **topological sort**, visits each node once, then traverses backward.

These are **two different reasons gradients add**:

1. Multiple paths **inside one computation graph** contribute to one parameter.
2. Multiple separate calls to `backward()` accumulate into existing `.grad` buffers until explicitly reset.

For four equal-sized micro-batches whose losses have already been averaged within each micro-batch, use $L_i/4$ before each backward, then update once. Unequal sizes require weighting by sample counts. BatchNorm and other batch-dependent operations can prevent exact equivalence to one full batch even with correct gradient arithmetic.

### 3.5. In-place modifications and graph lifetime

A first `backward()` typically frees saved values needed only for that backward. A second backward over the **same graph** may fail if those values are required; `retain_graph` can preserve them at a memory cost, but ordinary training should generally run a fresh forward for the next batch.

PyTorch tracks tensor version counters. If a backward operation saved a tensor, and its relevant contents are modified in-place before backward, PyTorch may throw a **version mismatch** rather than silently compute an incorrect derivative. A no-grad context does **not** make it safe to corrupt tensors that an earlier graph still needs.

### 3.6. Why activations dominate some training configurations

Saved activations are live tensors required to compute gradients during backward. For a particular non-recomputation Transformer implementation described in the course:

$$
\boxed{M_{\mathrm{activation}}\approx s b h L \left(34+\frac{5as}{h}\right)}
=34sbhL+5ab s^2 L\quad\text{bytes}.
$$

Definitions: sequence length $s$, micro-batch $b$, hidden width $h$, layers $L$, attention heads $a$. Coefficients are specific to the cited algorithm/precision assumptions. The attention-related term grows as **$s^2$**; the other term grows **linearly** with $s$.

For GPT-2-small-like values $s=1024$, $b=8$, $h=768$, $a=12$, $L=12$, the PDF estimates **~684 MiB per layer** of saved activations, or **~8.0 GiB across 12 layers**. In its illustrative comparison, that exceeds the parameter, gradient, and Adam/master-weight storage combined.

**Activation checkpointing / recomputation:** Keep selected layer inputs; during backward, recompute intermediate values rather than retaining them across the whole forward. A very simplified layer-boundary estimate is $2sbhL$ bytes, but **peak memory still includes the layer currently being recomputed and any temporary workspaces**. One extra forward can cost roughly one-third of a `6ND` training step in the basic compute model.

**Exam traps:** Autograd differentiates the *actual code*, even if its broadcast semantics are wrong; no-grad does not fix broken saved tensors; checkpointing trades memory for computation and does not delete all transient activation memory.

---
## 4. `nn.Module`, Parameters, Buffers, and Checkpoints

### 4.1. Forward execution is not the same as parameter registration

The source illustrates a common Python mistake: storing trainable layers inside an ordinary `list`. Those layers may still run in `forward()` and receive autograd gradients, but the parent's `model.parameters()` does **not** necessarily find them. An optimizer constructed from that list may therefore never update the hidden layers. Similarly, unregistered state may fail to move with `model.to(device)` or appear in a checkpoint.

In Python, `nn.Module` attribute management registers assigned `nn.Parameter` objects, child `nn.Module` objects, and explicitly registered buffers. `ModuleList`, `ModuleDict`, and `ParameterList` solve distinct registration problems; `Sequential` additionally defines a sequential forward composition. In C++/LibTorch, the analogous registration is **explicit** through `register_module`, `register_parameter`, and `register_buffer`.

```cpp
#include <torch/torch.h>

struct TinyNetImpl : torch::nn::Module {
    torch::nn::Linear fc{nullptr};
    torch::Tensor step_count;

    TinyNetImpl() {
        fc = register_module("fc", torch::nn::Linear(2, 1));
        step_count = register_buffer("step_count", torch::zeros({1}));
    }
    torch::Tensor forward(const torch::Tensor& x) {
        return fc->forward(x);
    }
};
TORCH_MODULE(TinyNet);

int main() {
    auto model = TinyNet();
    for (const auto& item : model->named_parameters()) {
        std::cout << item.key() << ": " << item.value().sizes() << "\n";
    }
}
```

This is intentionally a **C++ frontend** illustration; Python's `__setattr__`-based registration is not the C++ mechanism.

### 4.2. Parameters, buffers, and ordinary attributes

| State type                                      | Included in `parameters()`? | Included in ordinary model checkpoint? | Automatically moved with registered model? | Ordinary optimizer updates it?                               |
| ----------------------------------------------- | --------------------------- | -------------------------------------- | ------------------------------------------ | ------------------------------------------------------------ |
| Registered `Parameter`                          | Yes                         | Yes                                    | Yes                                        | Only if optimizer includes it and receives a usable gradient |
| Persistent registered buffer                    | No                          | Yes                                    | Yes                                        | No, unless updated by custom logic                           |
| Non-persistent registered buffer                | No                          | No                                     | Yes                                        | No                                                           |
| Plain tensor attribute (unregistered)           | No                          | No                                     | Not by recursive module state traversal    | No                                                           |
| Child module in unregistered ordinary container | Not via parent traversal    | Not via parent traversal               | Not via parent traversal                   | Not via parent-provided parameters                           |

**BatchNorm example:** `running_mean`, `running_var`, and `num_batches_tracked` are non-parameter state. Running statistics need to persist and move with the module; they are updated by training-forward statistics rather than ordinary optimizer gradients. A `(256,)` FP32 `running_mean` and `(256,)` FP32 `running_var` together occupy `2 × 256 × 4 = 2 KiB`, apart from the counter.

**Persistent vs non-persistent:** A recomputable RoPE lookup table or causal mask can be registered as a non-persistent buffer so it moves with the model but is rebuilt from configuration rather than serialized. The proper choice depends on reproducibility and whether the state can be regenerated exactly.

### 4.3. Count scalar parameters, not just tensor objects

`nn.Linear(512,2048)` with bias has:

$$
\boxed{512\times2048+2048=1{,}050{,}624\;\text{scalar parameters}.}
$$

It still contains **two Parameter tensors** (weight and bias). Total parameter count is the sum of `numel()` over independent registered parameters, accounting for shared/tied weights rather than counting tensor keys blindly.

For GPT-2-style tied embedding and output projection, a single parameter can be referred to under multiple attribute paths. A checkpoint may contain several names that refer to shared data; counting keys does not measure independent parameter storage.

### 4.4. `state_dict()` is not automatically a frozen snapshot

In Python PyTorch, calling `model.state_dict()` produces a mapping whose Tensor values generally **share** storage with live module state. If training changes a weight from 2 to 3 and you save a previously obtained shallow state mapping later, the saved value can be 3. To retain *the earlier version*, serialize immediately or make a genuine deep copy when the snapshot is taken.

Checkpoint keys reflect the registration hierarchy, e.g. `blocks.3.attn.qkv.weight`. Renaming `attn` to `attention` can make old keys incompatible. DDP model wrappers commonly introduce a `module.` prefix if the wrapper itself was serialized.

**`strict=False` is not an automatic name translator.** Missing and unexpected key lists must be inspected: it is possible to suppress the exception while loading **none** of the intended model weights. Shape mismatches are not universally ignored either.

### 4.5. Model versus optimizer state and device placement

A model checkpoint often contains parameter values and persistent buffers; a **resumable training checkpoint** may also require optimizer moments, LR schedule position, dynamic loss-scaling state, and training counters. Optimizer state is separate from module buffers.

The source recommends a practical setup order: **build model → move model to device → create optimizer → restore the appropriate checkpoint states**. Follow the framework-specific instructions if the restoration path has special constraints. Changing module parameters after constructing the optimizer may invalidate its references.

### 4.6. Freezing has three independent meanings

| Question                                               | Mechanism                                        | Why it matters                                 |
| ------------------------------------------------------ | ------------------------------------------------ | ---------------------------------------------- |
| Does this weight need derivatives?                     | `requires_grad_(false)` or equivalent            | Suppresses parameter gradients                 |
| Is this weight included in optimizer updates?          | Optimizer parameter groups and available `.grad` | Determines whether an update happens           |
| Does this layer behave in training or evaluation mode? | `train()` / `eval()`                             | Changes Dropout and typical BatchNorm behavior |

**Important:** Freezing a module's weights does not necessarily remove its backward path if its *input* needs a gradient. BN running statistics can still change during `train()` even when all BN parameters are frozen. Existing Adam moments also do not disappear just because a parameter is later frozen. The source's example: `3.5B` previously trained parameters with FP32 Adam `m` and `v` retain `3.5B×8 = 28 GB` of moments unless that state is explicitly released or reorganized.

**Exam traps:** “The layer ran” does not prove “the optimizer updates it”; “it is a buffer” does not imply “its value never changes”; `eval()` does not disable autograd; `strict=False` does not prove a checkpoint loaded correctly.

---

## 5. The Training Loop: Loss, SGD, AdamW, Scheduling, and Clipping

*Source: PDF pp. 50–62.*

### 5.1. One independent optimization step

For a scalar weight $w=1$, input $x=2$, target $y=0$, squared loss $L=(wx-y)^2$:

$$
\hat y=2,\quad L=4,\quad\frac{\partial L}{\partial w}=2(wx-y)x=8.
$$

With SGD learning rate $0.1$:

$$
w_{\mathrm{new}}=1-0.1\times8=0.2,\qquad L_{\mathrm{new}}=(0.2\times2)^2=0.16.
$$

**Forward computes values, backward computes gradients, optimizer step changes parameters.** None of these automatically clears previous gradients.

The usual conceptual order is:

1. Clear gradients (`optimizer.zero_grad`, often setting them to absent).
2. Run forward and compute loss.
3. Call `loss.backward()`.
4. Optionally unscale/clamp gradients; compute diagnostics.
5. Perform `optimizer.step()`.
6. Advance a scheduler according to its designated unit of *actual optimizer updates*, if applicable.

**C++/LibTorch training example (without Python-only AMP):**

```cpp
#include <torch/torch.h>

struct ClassifierImpl : torch::nn::Module {
    torch::nn::Linear fc{nullptr};
    ClassifierImpl() {
        fc = register_module("fc", torch::nn::Linear(8, 3));
    }
    torch::Tensor forward(const torch::Tensor& x) { return fc(x); }
};
TORCH_MODULE(Classifier);

int main() {
    auto model = Classifier();
    torch::optim::AdamW optimizer(
        model->parameters(),
        torch::optim::AdamWOptions(3e-4).weight_decay(0.01));

    for (int step = 0; step < 100; ++step) {
        auto x = torch::randn({16, 8});
        auto labels = torch::randint(0, 3, {16}, torch::kLong);
        optimizer.zero_grad();
        auto logits = model->forward(x);
        auto loss = torch::nn::functional::cross_entropy(logits, labels);
        loss.backward();
        torch::nn::utils::clip_grad_norm_(model->parameters(), 1.0);
        optimizer.step();
    }
}
```

Cross-entropy expects appropriate **logits** and class targets; it is not the same as manually averaging logits across the class axis. Uniform predictions over `n` classes give initial loss $-\ln(1/n)=\ln n$ **only under the uniform-prediction assumption**.

### 5.2. Intentional gradient accumulation

For `ACC=4` equal-size micro-batches, each returning its own **mean loss**, use:

$$
\boxed{g_{\mathrm{effective}}=\sum_{k=1}^{4}\nabla_\theta\left(\frac{L_k}{4}\right)}.
$$

If their mean gradients are `2, 4, 6, 8`, correctly normalized accumulation yields `(2+4+6+8)/4=5`, not `20`. Do not step the optimizer until the accumulation window is complete. For unequal micro-batch sizes $n_k$, a mean-of-samples objective uses weights $n_k/\sum_j n_j$.

```cpp
#include <torch/torch.h>
// Call with a forward function that returns logits requiring gradients,
// an optimizer covering the forward model's trainable parameters,
// and an accumulation group of equal-sized micro-batches.
template <typename Forward>
void accumulated_step(
    Forward&& forward,
    torch::optim::Optimizer& optimizer,
    const std::vector<std::pair<torch::Tensor, torch::Tensor>>& batches) {
    TORCH_CHECK(!batches.empty(), "At least one micro-batch is required");
    optimizer.zero_grad();
    for (const auto& [input, labels] : batches) {
        auto logits = forward(input);
        auto loss = torch::nn::functional::cross_entropy(logits, labels)
                  / static_cast<double>(batches.size());
        loss.backward();
    }
    optimizer.step();
}
```

This template receives a caller-provided C++ forward callable rather than a hard-coded model class, allowing the **accumulation arithmetic** to operate on a real differentiable model. Final incomplete windows need their actual sample counts; BatchNorm can make micro-batch processing differ from one large batch.

### 5.3. `None` gradient versus a zero-valued gradient

- Absent gradient (`grad is None`) means there is no gradient tensor for this update; ordinary optimizers commonly **skip** that parameter.
- Zero-valued gradient means a gradient tensor exists; momentum and weight decay can still affect the update.
- `zero_grad(set_to_none=True)` can relinquish gradient buffers to the allocator for reuse; it does not imply `nvidia-smi` usage drops immediately.

The PDF's SGD-with-momentum example: parameter starts at `1`, first gradient `2`, LR `0.1`, momentum `0.9`. After the first step, parameter is `0.8`, momentum buffer `2.0`. A following step with **no gradient** keeps `0.8` and `2.0`; a step with an **explicit zero gradient** updates momentum to `1.8` and parameter to **`0.62`**. This distinction matters when freezing/unfreezing parameters.

### 5.4. Adam and AdamW

For Adam, at update $t$, with gradient $g_t$:

$$
\begin{aligned}
m_t &= \beta_1m_{t-1}+(1-\beta_1)g_t,\\
v_t &= \beta_2v_{t-1}+(1-\beta_2)g_t^2,\\
\hat m_t &= \frac{m_t}{1-\beta_1^t},\qquad
\hat v_t=\frac{v_t}{1-\beta_2^t},\\
\theta_{t+1} &=\theta_t-\eta\frac{\hat m_t}{\sqrt{\hat v_t}+\varepsilon}.
\end{aligned}
$$

Typical illustrative settings: $\beta_1=0.9$, $\beta_2=0.999$, $\varepsilon=10^{-8}$. For first gradient $g=2$ and zero moments, $m_1=0.2$, $v_1=0.004$, $\hat m_1=2$, $\hat v_1=4$. With $\eta=3\times10^{-4}$, the first gradient-driven update magnitude is approximately $3\times10^{-4}$, provided epsilon is negligible.

**Weight decay distinction:**

| Coupled L2-like Adam treatment                       | Decoupled AdamW                                              |
| ---------------------------------------------------- | ------------------------------------------------------------ |
| Feed $g+\lambda\theta$ into first and second moments | Keep moments based on the data gradient $g$                  |
| Regularization affects adaptive moment history       | Separately shrink weights by factor $(1-\eta\lambda)$        |
| Not equal to uniform parameter shrinkage             | $\theta\leftarrow\theta(1-\eta\lambda)$ plus adaptive gradient update |

For decay-only step with $\theta=2$, LR `0.01`, decay `0.1`, AdamW shrinkage gives `2×(1−0.001)=1.998` before applying the data-gradient update. This does **not** mean decay has no learning-rate dependence. Source and official docs agree that AdamW decouples decay from moment statistics.

### 5.5. Learning-rate schedule: define what a “step” means

Warmup raises LR gradually, often followed by cosine decay. The PDF describes illustrative warmup windows of `0.5%–2%` of total scheduled updates—not a universal recommendation. The sample schedule ends at `10%` of its peak LR, which is also not mandatory.

**Clock mismatch example:** `10 epochs × 100 batches = 1,000 batches`. A scheduler configured for **10 epoch-level calls** but called **every batch** advances 100 times more frequently than intended. With accumulation, decide whether the LR schedule advances for every micro-batch, every attempted update, or every *successful* optimizer update. AMP overflow skip decisions may matter.

### 5.6. Gradient clipping and why order matters

**Global L2 norm clipping:** For gradients `[3,4]`, norm is `5`. To cap norm at `1`, multiply all elements by approximately `1/5`:

$$
\boxed{[3,4]\ \longrightarrow\ [0.6,0.8]}
$$

The direction is preserved. **Value clipping** at ±1 instead produces `[1,1]`, changing direction. Clip after backward and after all accumulation; **when using loss scaling, unscale first**, then clip, then step. A useful diagnostic is the **pre-clipping norm**, which `clip_grad_norm_` commonly returns.

### 5.7. `train()`, `eval()`, no-grad, and inference mode

| Operation                          | What it controls                                             | What it does *not* guarantee                                 |
| ---------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| `model.train()`                    | Training-dependent module behavior                           | Does not force every parameter to require gradients          |
| `model.eval()`                     | Eval-dependent behavior (e.g., Dropout, BatchNorm)           | Does not disable autograd                                    |
| `NoGradGuard` / Python `no_grad()` | Suppresses ordinary backward graph recording for operations in scope | Does not set BatchNorm to evaluation mode                    |
| Python `inference_mode()`          | Stricter inference context, reduced tracking overhead        | Not a universal replacement for data that must later participate in autograd |
| `requires_grad_(false)`            | Which parameters request gradients                           | Does not fix BatchNorm running statistics                    |

```cpp
#include <torch/torch.h>
// model is a registered LibTorch model on the chosen device.
void evaluate(const torch::nn::Module& /*model*/) {
    torch::NoGradGuard no_grad;  // Guards operations within this scope.
    // Call model->eval() on a non-const model before evaluation.
    // Then run forward and compute metrics without building a backward graph.
}
```

**GPU logging caveat:** Extracting `loss.item()` on every iteration can synchronize the CPU with pending GPU work. The PDF recommends accumulating **detached** losses on the GPU and transferring a scalar every ~50 steps *when appropriate*. Always count the actual number of terms in the last partial window; never accumulate a loss tensor that inadvertently preserves unwanted graph references.

**Exam traps:** An optimizer step does not clear gradients; Adam's first update is not always exactly `±LR` in the presence of epsilon/decay; `eval()` and no-grad solve different problems; clipping scaled gradients changes the effective threshold.

---

## 6. Mixed Precision: FP16, BF16, TF32, Loss Scaling

*Source: PDF pp. 63–74.*

### 6.1. Floating-point formats: range versus resolution

A floating-point representation encodes sign, exponent, and significand. **Exponent width controls the range of magnitudes; mantissa width controls spacing between nearby representable values.**

| Format   | Sign | Exponent | Fraction bits | Relative spacing near 1 (normal numbers) | Notes                                         |
| -------- | ---- | -------- | ------------- | ---------------------------------------- | --------------------------------------------- |
| FP32     | 1    | 8        | 23            | $2^{-23}\approx1.19\times10^{-7}$        | 4-byte storage                                |
| FP16     | 1    | 5        | 10            | $2^{-10}\approx9.77\times10^{-4}$        | 2-byte storage, narrower range                |
| BF16     | 1    | 8        | 7             | $2^{-7}\approx7.81\times10^{-3}$         | 2-byte storage, FP32-like exponent range      |
| TF32     | 1    | 8        | 10            | Compute-format precision                 | Does **not** make FP32 tensors occupy 2 bytes |
| FP8 E4M3 | 1    | 4        | 3             | Very coarse                              | Typically needs explicit scale management     |

**FP16 thresholds to memorize:**

$$
\begin{aligned}
\text{Largest finite FP16} &=65504,\\
\text{Smallest positive normal FP16} &=2^{-14}\approx6.10\times10^{-5},\\
\text{Smallest positive subnormal FP16} &=2^{-24}\approx5.96\times10^{-8}.
\end{aligned}
$$

**Nuance:** Numbers smaller than the smallest *normal* FP16 do not all become zero: **subnormal values** exist, though exact behavior can depend on the arithmetic implementation and rounding. The source's examples: `1e-8` rounds to zero in ordinary FP16 representation, whereas `3e-8` may round up to the first subnormal. For BF16, the exponent range resembles FP32 but the representable boundaries and resolution are not identical.

**Two failure modes:**

1. **Underflow/overflow:** A gradient may be too small to represent or a value may exceed the format's finite range.
2. **Lost small updates:** A nonzero change may round away when added to a much larger low-precision parameter. Around FP16 value `1`, the next higher representable number is separated by approximately `0.0009765625`; BF16 spacing is coarser.

### 6.2. Why loss scaling works and what it cannot fix

Loss scaling computes $L'=S L$ before backward, so gradients become:

$$
\frac{\partial L'}{\partial\theta}=S\frac{\partial L}{\partial\theta}.
$$

If a gradient was `1e-8`, scaling by `1024` makes it `1.024e-5`, which can remain nonzero in FP16 **while it is being represented**. Before optimizer updates, divide the gradient by `1024` in a suitable higher-precision representation.

**Critical distinction:** Scaling must happen **before** data would be lost. Multiplying a value *after it has rounded to zero* cannot restore information. Scales that are too large can themselves create overflow in intermediate gradients.

### 6.3. Dynamic loss scaling in the Python AMP workflow

The book's Python `GradScaler` example starts at scale `65536`, with growth factor `2`, backoff `0.5`, and growth interval `2000` (example settings). Its conceptual workflow is:

1. Scale the loss and run backward.
2. Unscale the optimizer's gradients before inspecting/clipping them.
3. Detect `inf`/`nan`; skip an unsafe optimizer step.
4. Update the scale according to the overflow history.

**C++ note:** `torch.amp.autocast` and `GradScaler` are **Python frontend API names used in the textbook**. This review does not invent identical LibTorch C++ classes or pretend those Python calls are C++. The following pure C++ excerpt explains the numerical principle only:

```cpp
#include <cmath>
#include <iostream>
int main() {
    const double original_gradient = 1e-8;
    const double scale = 1024.0;
    const double scaled_gradient = original_gradient * scale;
    const double recovered_gradient = scaled_gradient / scale;
    std::cout << scaled_gradient << " -> " << recovered_gradient << "\n";
    // Real AMP must scale before low-precision backward math and check overflow.
}
```

**Do not misdiagnose `131072` as automatic FP16 overflow in all contexts.** A scaled scalar loss of `2×65536=131072` can be perfectly finite when held in FP32, as is common for cross-entropy losses under AMP. Inspect the actual dtype and the intermediate gradients.

### 6.4. FP32 master weights and autocast solve different problems

- **Loss scaling:** preserves small intermediate backward gradients from FP16 underflow.
- **FP32 master weight / FP32 model parameter:** accumulates small parameter updates that would repeatedly round away in FP16 or BF16.
- **FP32 accumulation:** improves accumulation within certain operations, but cannot recover already rounded low-precision inputs.
- **Autocast:** selects compute dtype **per eligible operator** (and hardware rules), not a blanket conversion of the entire model to half precision.

A common native AMP setup retains FP32 parameter and gradient storage, while some forward operators execute at lower precision. Another explicit-master-copy design stores both low-precision working weights and FP32 master weights. These two designs may both total **16 bytes per parameter** for parameter/gradient/moment-related states under the textbook's assumptions:

| Design                                                       | Steady-state component bytes per parameter | Total    |
| ------------------------------------------------------------ | ------------------------------------------ | -------- |
| Native AMP-like FP32 params/gradients + FP32 Adam moments    | `4 + 4 + 4 + 4`                            | **16 B** |
| Explicit BF16 copies + BF16 gradients + FP32 master + FP32 moments | `2 + 2 + 4 + 4 + 4`                        | **16 B** |

These totals exclude temporarily cast parameters, activation buffers, workspaces, fragmentation, etc. **“Half-precision model” is not synonymous with “half the total training memory.”**

### 6.5. FP16 vs BF16 vs TF32 selection

**FP16:** More fraction bits than BF16, but a limited exponent range. Often paired with loss scaling for certain training workloads.  
**BF16:** Wider exponent range, fewer fraction bits, commonly does not need FP16-style dynamic loss scaling. It can still produce `inf`, `nan`, or loss of small updates.  
**TF32:** Uses FP32 storage while allowing reduced internal multiplication precision on compatible Tensor Core paths. It is not a 16-bit tensor storage dtype.  
**FP8:** Often requires per-tensor/per-channel/block scaling, finite-range management, and appropriate hardware/software support.

**Hardware numerical illustration from the PDF (A100):** FP32 `19.5 TFLOPS`, TF32 Tensor Core `156 TFLOPS`, and BF16/FP16 Tensor Core `312 TFLOPS` under their respective specification assumptions. Peak FLOPS ratios **are not** predicted end-to-end training speedups.

**Exam traps:** BF16's range does not mean high precision; loss scaling does not fix lost parameter updates; `model.half()` does not automatically implement safe mixed precision; clipping belongs **after** AMP unscale.

---
## 7. GPU Memory Accounting: Parameters, Gradients, Optimizer State, Activations

*Source: PDF pp. 75–86.*

### 7.1. The four major memory categories

An inference model whose weights need `14 GB` can require vastly more memory for **full-parameter training**. Training storage includes:

| Category        | What it stores                                               | Primary scaling                                     |
| --------------- | ------------------------------------------------------------ | --------------------------------------------------- |
| Parameters      | Model weights (possibly multiple precision copies)           | Parameter count $N$                                 |
| Gradients       | Backpropagated derivatives                                   | Trainable parameter count                           |
| Optimizer state | Adam moments, master weights, step counters, etc.            | Updated parameter count / optimizer                 |
| Activations     | Forward values retained for backward, plus live recomputation/workspaces | Micro-batch, sequence, architecture, implementation |

**Important temporal detail:** Gradients can appear during backward; Adam moments can be allocated lazily on the **first optimizer update**. Measuring only immediately after model loading can miss most of the eventual peak.

### 7.2. Bytes per parameter: specify the precision and optimizer assumptions

| Example training plan                                        | Components                                                   | Approximate bytes/parameter                             |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------- |
| FP32 + Adam/AdamW                                            | FP32 parameter `4` + FP32 gradient `4` + FP32 `m` `4` + FP32 `v` `4` | **16 B**                                                |
| Explicit BF16 working copy with FP32 master + FP32 Adam      | BF16 weights `2` + BF16 gradients `2` + FP32 master `4` + FP32 `m` `4` + FP32 `v` `4` | **16 B**                                                |
| FP32 SGD with momentum                                       | FP32 weights `4` + gradients `4` + momentum `4`              | **12 B**                                                |
| Illustrative BF16 weights/gradients with FP32 moments, no separate master | `2 + 2 + 4 + 4`                                              | **12 B**                                                |
| Illustrative 8-bit Adam moments with explicit master         | `2 + 2 + 4 + 1 + 1`                                          | **10 B**, *before* quantization metadata and exceptions |

**Do not equate “FP32 optimizer-related state 12 B” with “two Adam moments 12 B”.** In the explicit-master design, **master + `m` + `v` = `4+4+4=12 B`; `m` + `v` alone = `8 B`**.

The simple estimate is:

$$
\boxed{M_{\mathrm{parameter\ state}}\approx N\times\mathrm{bytes\ per\ parameter}}.
$$

### 7.3. Model size calculation

Under the 16 B/parameter textbook design (not counting activations):

| Model | Arithmetic             | Result (decimal GB) |
| ----- | ---------------------- | ------------------- |
| 7B    | $7\times10^9\times16$  | **112 GB**          |
| 13B   | $13\times10^9\times16$ | **208 GB**          |
| 70B   | $70\times10^9\times16$ | **1,120 GB**        |

For a 7B model under the explicit BF16+master scheme: BF16 working weights `14 GB`, BF16 gradients `14 GB`, and FP32 master plus two FP32 moments `84 GB` = `112 GB` before any activation tensors or transient allocations. **Reducing micro-batch size to one does not remove this parameter-state requirement.**

**GB vs GiB:** $1\,\mathrm{GB}=10^9$ bytes; $1\,\mathrm{GiB}=2^{30}$ bytes. A specification showing `81920 MiB` represents `80 GiB ≈ 85.90 GB` before device reservation and runtime overhead.

### 7.4. Worked activation estimate for a 7B-like architecture

From the source's selected model assumptions:

| Parameter           | Value  |
| ------------------- | ------ |
| Hidden width $h$    | `4096` |
| Layers $L$          | `32`   |
| Attention heads $a$ | `32`   |
| Sequence length $s$ | `4096` |
| Micro-batch $b$     | `1`    |

The course's **naïve attention saved-activation model** is:

$$
M_{\mathrm{act}}\approx sbhL\left(34+\frac{5as}{h}\right).
$$

Substitute:

$$
\begin{aligned}
sbhL &=4096\times1\times4096\times32=536{,}870{,}912,\\
34+5as/h &=34+5\times32=194,\\
\boxed{M_{\mathrm{act}}} &\boxed{\approx104.2\;\mathrm{GB}}.
\end{aligned}
$$

This huge estimate is driven partly by explicitly saved attention-related tensors with $s^2$ dependence. It **does not mean all implementations of this architecture use 104 GB of activations**.

**Simplified memory-efficient-attention comparison:** Drop only the $5as/h$ quadratic term and keep coefficient `34`:

$$
M_{\mathrm{approx,no\ full\ scores}}=34sbhL\approx18.3\;\mathrm{GB}.
$$

This is an **illustration**, not a precise FlashAttention allocation model; FlashAttention has its own saved data and workspaces. A boundary-only activation checkpointing estimate is $2sbhL\approx1.07\,\mathrm{GB}$, **not** the total recomputation-time peak.

**Sequence length sensitivity from the PDF (same $h,L,a,b$):**

| Sequence length | Naïve activation estimate | “Remove quadratic term” estimate | Saved layer-input boundaries only |
| --------------- | ------------------------- | -------------------------------- | --------------------------------- |
| 1,024           | 9.9 GB                    | 4.6 GB                           | 0.27 GB                           |
| 2,048           | 30.6 GB                   | 9.1 GB                           | 0.54 GB                           |
| 4,096           | 104.2 GB                  | 18.3 GB                          | 1.07 GB                           |
| 8,192           | 380.1 GB                  | 36.5 GB                          | 2.15 GB                           |

This table illustrates how the quadratic attention component eventually dominates. With GQA, the number of KV heads may decrease, **but the explicit query-head score tensor** can remain of shape `(batch, query_heads, seq, seq)` if the implementation still materializes it.

**C++ training-memory estimator** (arithmetic only; no GPU required):

```cpp
#include <cstdint>
#include <iomanip>
#include <iostream>

struct EstimateGB { double naive, no_quadratic, boundaries; };

EstimateGB estimate_activation(int64_t seq, int64_t micro_batch = 1,
                               int64_t hidden = 4096, int64_t layers = 32,
                               int64_t heads = 32) {
    const double base = double(seq) * micro_batch * hidden * layers;
    const double coefficient = 34.0 + 5.0 * heads * seq / hidden;
    return {base * coefficient / 1e9,
            base * 34.0 / 1e9,
            base * 2.0 / 1e9};
}

int main() {
    for (int64_t s : {1024, 2048, 4096, 8192}) {
        auto m = estimate_activation(s);
        std::cout << s << ": " << std::fixed << std::setprecision(2)
                  << m.naive << " GB | " << m.no_quadratic
                  << " GB | " << m.boundaries << " GB\n";
    }
    // These are model estimates; do not mistake them for measured CUDA peaks.
}
```

### 7.5. A ZeRO-1-style shard calculation

For a 13B model and **8-way optimizer-state sharding** under the book's explicit-master assumptions:

- Full BF16 weights and BF16 gradients remain on each rank: $13\times10^9\times(2+2)=52\,\mathrm{GB}$.
- FP32 master weights and Adam `m`,`v` are sharded: $13\times10^9\times12/8=19.5\,\mathrm{GB}$ per rank.
- Total parameter-related estimate per rank: **`71.5 GB`**, not `208/8 = 26 GB`.

This is one simplified partitioning pattern; ZeRO-2, ZeRO-3, FSDP, activation sharding, all-gather temporary buffers, and communication overlap create different per-rank behavior. **Always state exactly which components are sharded.**

### 7.6. `allocated`, `reserved`, visible VRAM, and fragmentation

| Observation              | Meaning                                                      | What not to conclude                             |
| ------------------------ | ------------------------------------------------------------ | ------------------------------------------------ |
| `memory_allocated()`     | Memory occupied by active PyTorch-managed tensor allocations | Not all process or driver GPU memory             |
| `memory_reserved()`      | Memory managed by PyTorch's CUDA caching allocator           | Not all currently live tensors                   |
| `reserved - allocated`   | Allocator-managed but not currently allocated to active tensors | Not automatically a memory leak or fragmentation |
| `nvidia-smi` usage       | Broad device-visible usage including other overheads         | Not identical to either PyTorch counter          |
| `max_memory_allocated()` | Peak allocated within measurement window                     | Not specifically activation memory               |

The source OOM message says: **request 2.00 GiB**, **driver free 1.42 GiB**, **PyTorch allocated 70.21 GiB**, and **PyTorch reserved-but-unallocated 5.32 GiB**. The simple fact `1.42 < 2.00` already matters. The additional 5.32 GiB requires allocator-state investigation; it is **not enough evidence to diagnose fragmentation**.

Under a simplified allocator example, seven free units in disjoint blocks of lengths `1,2,1,3` cannot satisfy a **single four-unit contiguous block** without other allocator actions. This is a conceptual fragmentation example, not a claim that GPU physical VRAM is one continuous linear array.

The official CUDA semantics documentation clarifies that **`empty_cache()` releases unused cached allocations to other GPU applications; it does not free tensors that are still live or solve the model's fundamentally required memory.** On newer PyTorch documentation, the allocator tuning environment variable is named `PYTORCH_ALLOC_CONF`, with `PYTORCH_CUDA_ALLOC_CONF` retained as an alias. Options like `expandable_segments` or `max_split_size_mb` should be justified by memory history and backend support, **not applied to every OOM by default**.

### 7.7. Practical OOM triage

1. Record GPU/model/dtype, parameter count, micro-batch, sequence length, and optimizer.
2. Compute parameter-related lower bound **before** changing random settings.
3. Measure *after loading*, *after first forward*, *after backward*, *after first optimizer step*, and at representative steady-state steps.
4. Record current allocated, peak allocated, reserved, and global free/used GPU memory.
5. If **parameter states** dominate: consider sharding, optimizer-state compression, offload, or a parameter-efficient tuning method when task requirements permit.
6. If **activations** dominate: consider lower micro-batch plus correct gradient accumulation, recomputation, and memory-efficient attention.
7. If evidence points to **allocator fragmentation**: inspect memory snapshots before changing allocator settings.
8. Recheck numerical correctness and wall-clock throughput after a memory optimization; offload, recomputation, and sharding all have potential time costs.

**Exam traps:** 112 GB static state remains 112 GB even with batch size 1; peak GPU memory may occur during backward or the first optimizer step; buffer caching is not necessarily leakage; the boundary-only recomputation curve is not an end-to-end peak.

---

## 8. CUDA Asynchrony, Pinned Memory, DataLoader, and Profiling

*Source: PDF pp. 87–99.*

### 8.1. Three separate stages in a batch

A typical input-to-computation path:

**Storage/file read → CPU decode/transform → batch collation → interprocess transfer → optional pinning → host-to-device (H2D) copy → GPU kernels.**

CPU preprocessing, H2D copying, and GPU compute are **different** activities; they can overlap across distinct batches when hardware, buffering, synchronization, and scheduling permit.

**Source's timeline illustration:** Each batch requires CPU prep `10 ms`, H2D `4 ms`, and GPU compute `12 ms`.

- Fully serial: $3\times(10+4+12)=\boxed{78\,\mathrm{ms}}$ for three batches.
- Under the source's assumed prefetch/overlap schedule: first batch arrives at `14 ms`, then three compute intervals take `36 ms` without an intervening stall: **`50 ms`** to complete three batches.

This **does not** imply any `DataLoader(pin_memory=true)` configuration automatically produces the 50 ms timeline. Data preparation, asynchronous copy, **separate streams**, and correct dependencies all matter.

### 8.2. CPU enqueue versus GPU completion

PyTorch CUDA operations commonly enqueue work and return control to the host **before the GPU has finished**. Tensor metadata can exist while the actual tensor values are still being computed. Dependencies on the same CUDA stream preserve necessary ordering.

**Implicit synchronization can occur when the host needs actual GPU values:** a scalar conversion such as Python `loss.item()`, printing a GPU tensor, device-to-host reads, or a host-side conditional involving a GPU scalar.

**Bad benchmark:** Wall-clock time only around an asynchronous kernel-launch call, without waiting for completion.  
**Better benchmark:** Warm up, synchronize before starting, enqueue a representative run, synchronize before stopping, repeat, and report variability. CUDA events can measure time between two device-side markers (including waiting between them); they do not automatically measure only arithmetic work.

**C++/LibTorch synchronization benchmark** (CUDA-enabled LibTorch build needed):

```cpp
#include <torch/torch.h>
#include <torch/cuda.h>
#include <chrono>
#include <iostream>

int main() {
    TORCH_CHECK(torch::cuda::is_available(), "CUDA required");
    auto a = torch::randn({2048, 2048}, torch::kCUDA);
    auto b = torch::randn({2048, 2048}, torch::kCUDA);
    for (int i = 0; i < 5; ++i) { auto warm = a.matmul(b); }
    torch::cuda::synchronize();

    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < 20; ++i) { auto result = a.matmul(b); }
    torch::cuda::synchronize();
    auto t1 = std::chrono::steady_clock::now();
    const double ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count() / 20;
    std::cout << "Synchronized average: " << ms << " ms/iteration\n";
}
```

The loop above is illustrative. **Compiler/optimizer lifetime and allocation effects can influence microbenchmarks**; for rigorous kernel measurement use appropriate device events, a known working set, and explicit timing boundaries. `CUDA_LAUNCH_BLOCKING=1` can help localize errors, but it changes execution and should not be left on for ordinary performance comparison.

### 8.3. `non_blocking` and CUDA streams do different things

- `non_blocking=True` on a supported tensor transfer can reduce **host-side waiting** after copy submission.
- A **stream** is an ordered device work queue; independent streams can overlap eligible work, subject to hardware resources and dependencies.
- Copying **the current batch** and then computing on it still has a necessary copy-before-use dependency.
- The usual opportunity is **copying the next batch** while the GPU computes the current batch, using a suitable separate copy stream, events/waits, and correct buffer lifetime tracking.
- PyTorch Python `record_stream()` helps its caching allocator track cross-stream use. On the C++ side, the exact APIs depend on the frontend/ATen/CUDA interoperability layer; do not omit dependency or lifetime logic when implementing a real prefetcher.

**Pinned host buffer safety:** Do not mutate or release pinned source storage until an outstanding asynchronous H2D transfer has completed. For asynchronous GPU-to-CPU transfers, do not inspect the CPU result before the relevant copy is known complete.

### 8.4. Pinned memory: transfer benefits versus pinning cost

**Pageable memory** can be relocated/swapped by the operating system. **Pinned (page-locked) host memory** stays resident and can support more efficient DMA transfers. Yet producing a pinned copy has CPU/alloc/copy costs of its own.

The source quotes a **specific official PyTorch tutorial experiment**, not a universal hardware benchmark: many large tensor transfers (~4 GB total) were reported at approximately **391 ms** for blocking pageable copies, **313 ms** for non-blocking pageable copies, **300 ms** for already-pinned non-blocking copies, and **562 ms** for a timed path that *first pins and then performs blocking copies*. The last option is slower precisely because preparing pinned buffers is part of its critical path.

**Verified clarification:** Official PyTorch guidance emphasizes that a separate pinning stage managed by the data pipeline can be beneficial, whereas manual `pin_memory().to(device, non_blocking=True)` done on the hot path can be slower than a direct transfer. Your own result depends on tensor sizes, CPU cores, PCIe topology, DMA engines, streams, and competing memory pressure.

### 8.5. DataLoader knobs: capacity vs. buffer depth

| Setting                 | Primary effect                                      | Possible cost                                                |
| ----------------------- | --------------------------------------------------- | ------------------------------------------------------------ |
| `num_workers`           | Number of concurrent CPU sampling/transform workers | CPU contention, process startup, IPC overhead                |
| `prefetch_factor`       | Number of batches prepared/queued per worker        | Host/shared/pinned memory consumption                        |
| `pin_memory`            | Prepare page-locked host tensors                    | CPU pin/copy and additional host memory                      |
| `persistent_workers`    | Keep workers alive across epochs                    | Persistent worker datasets and state; epoch update coordination |
| `non_blocking` transfer | Avoid unnecessary per-transfer host waits           | Requires correct lifetime/synchronization handling           |

**Idealized worker sizing (source illustration):** One worker needs `40 ms` per batch; GPU needs `12 ms` compute per batch. To avoid *average* preprocessing starvation under ideal parallelism:

$$
\frac{W}{0.040}\geq\frac{1}{0.012}
\quad\Longrightarrow\quad
\boxed{W\geq\frac{40}{12}\approx3.33}.
$$

So **four workers** is a reasonable starting candidate *under these simplifying assumptions*, not a magic value. Idealized GPU ceiling: `1000/12≈83.3 batches/s`. Two workers supply at most `2×1000/40=50 batches/s`. Increasing prefetch depth can absorb short spikes, **but cannot fix a permanently lower average supply rate**. With `num_workers=4` and `prefetch_factor=2`, approximately `8` batches may be pre-queued, before counting other simultaneous buffers.

**Shared-memory pitfall:** Linux multi-process workers may exchange tensors through `/dev/shm`; Docker containers can have a small default shared-memory allocation in some environments (the source mentions `64 MB`). A worker `Bus error` may originate here rather than from GPU VRAM. Verify actual system limits before changing container settings.

### 8.6. Profiling bottlenecks instead of guessing

A productive debugging sequence is:

1. **Bypass the real data pipeline** using a correctly shaped pre-created GPU batch and measure a comparable training step. If that improves throughput, inspect input pipeline stages rather than asserting the GPU kernels are already optimal.
2. Inspect **GPU idle intervals** in Nsight Systems or the PyTorch profiler timeline; align them with host `DataLoader` waits, copies, synchronizations, and scheduling.
3. Distinguish `cudaStreamSynchronize` waits introduced by necessary dependencies from avoidable scalar logging/host sync.
4. If many tiny kernels create launch gaps, investigate fusion or compilation and measure the **whole step** after changes.
5. Check end-to-end throughput, peak VRAM, numerical correctness, and reproducibility—not just `nvidia-smi` GPU-Util.

`nvidia-smi` utilization estimates how often at least a kernel is executing in the sampling window. It does **not** measure the fraction of peak model FLOPS achieved (MFU). A single small long-running kernel can keep utilization high while useful mathematical throughput remains poor.

### 8.7. C++ concurrency experiment: why producer–consumer overlap helps

This example is **pure C++**, not a direct replacement for PyTorch's multi-process DataLoader. It demonstrates the general concept: let one producer prepare the next batch while the consumer processes the current batch.

```cpp
#include <chrono>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <queue>
#include <thread>

int main() {
    constexpr int kBatches = 8;
    std::queue<int> ready;
    std::mutex mutex;
    std::condition_variable cv;
    bool finished = false;

    auto producer = std::thread([&] {
        for (int i = 0; i < kBatches; ++i) {
            std::this_thread::sleep_for(std::chrono::milliseconds(40));
            {
                std::lock_guard<std::mutex> lock(mutex);
                ready.push(i);
            }
            cv.notify_one();
        }
        {
            std::lock_guard<std::mutex> lock(mutex);
            finished = true;
        }
        cv.notify_all();
    });

    auto begin = std::chrono::steady_clock::now();
    int consumed = 0;
    while (consumed < kBatches) {
        std::unique_lock<std::mutex> lock(mutex);
        cv.wait(lock, [&] { return !ready.empty() || finished; });
        if (ready.empty()) break;
        ready.pop();
        lock.unlock();
        std::this_thread::sleep_for(std::chrono::milliseconds(12));
        ++consumed;
    }
    producer.join();
    const auto elapsed = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - begin).count();
    std::cout << "Consumed " << consumed << " batches in " << elapsed << " ms\n";
}
```

For the source's simplified *one-worker* ideal model, preparation produces a batch every 40 ms and compute needs 12 ms. Eight batches take roughly `320 ms` to prepare, plus the final `12 ms` to compute, or **~332 ms**, ignoring scheduling overhead. The fully serial comparison is `8×(40+12)=416 ms`. With several independent workers the ideal model can approach the compute-limited throughput, but real CPU contention and IPC reduce gains.

**Exam traps:** Host nonblocking does not imply current-batch copy/compute overlap; the first batch still has startup latency; pinned memory is not free; more workers can become slower; perf traces need per-stage interpretation, not only a ranked list of expensive functions.

---
## 9. Reading Roadmap and External References

*Source: PDF pp. 100–106, with additional verification against official documentation.*

The book's ninth lesson is a **resource guide**, not an additional model mechanism. Its main rule: start from a concrete question, read only enough to resolve it, and return to a small reproducible test.

| Question you need to answer                           | First useful reference                                       | What to verify                                               |
| ----------------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| How do multiple gradients add?                        | [Karpathy: Neural Networks Zero to Hero](https://github.com/karpathy/nn-zero-to-hero) and [micrograd](https://github.com/karpathy/micrograd) | Work out $z=xy+x$ and confirm $dz/dx=y+1$                    |
| What does a tensor actually point to?                 | [ezyang: PyTorch Internals](https://blog.ezyang.com/2019/05/pytorch-internals/) | Draw storage and stride for a column slice; treat the 2019 implementation details as historical |
| Why did pinning or nonblocking copying not help?      | [Official pinned-memory and nonblocking guide](https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html) | Compare whether pinning cost and transfer completion are inside the measurement |
| What precision should this workload use?              | [PyTorch AMP guide](https://docs.pytorch.org/docs/stable/amp.html) and [AMP examples](https://docs.pytorch.org/docs/stable/notes/amp_examples.html) | Check hardware support, per-op dtype, scaler, and unscale-before-clip |
| How much training memory is needed?                   | [Transformer Math 101](https://blog.eleuther.ai/transformer-math/) | Write separate state/activation estimates and their assumptions |
| Why does an optimizer step behave this way?           | [PyTorch AdamW reference](https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html) | Verify moment updates and decoupled decay                    |
| Why is my gradient wrong after an in-place op?        | [Autograd mechanics](https://docs.pytorch.org/docs/stable/notes/autograd.html) | Saved tensors, version counters, leaf versus non-leaf        |
| How do I reason about CUDA memory and timing?         | [CUDA semantics](https://docs.pytorch.org/docs/stable/notes/cuda.html) | Async execution, caching allocator, reserved versus allocated |
| How does native LibTorch represent the same concepts? | [PyTorch C++ frontend](https://docs.pytorch.org/cppdocs/frontend.html) | C++ API availability and explicit module registration        |
| How are full training loops assembled?                | [nanoGPT](https://github.com/karpathy/nanoGPT)               | Trace accumulation, precision, schedule, optimizer state, and checkpoints |
| How do I debug training correctness?                  | [Karpathy: A Recipe for Training Neural Networks](https://karpathy.github.io/2019/04/25/recipe/) | Sanity-check labels, loss scale, and tiny-set overfitting    |
| Where do I find authoritative API signatures?         | [Official PyTorch tutorials and docs](https://docs.pytorch.org/tutorials/) | Match the actual installed PyTorch version and device        |

The PDF additionally recommends MartinLwx's tensor-stride walkthrough, Max Shapp's memory-estimation article, Sebastian Raschka's *Machine Learning with PyTorch and Scikit-Learn* and *Build a Large Language Model (From Scratch)*, and the PyTorch forum/source tree. These supplement the same core subjects rather than replacing direct experiments.

### 9.1. Verified clarifications added to the source notes

**Verified from official documentation — Tensor and C++ API:** The C++ frontend provides tensors, autograd, `torch::nn`, optimizers, data utilities, and serialization, but some Python training conveniences have distinct C++ APIs or no identical public interface. This is why this document avoids relabeling Python AMP or DataLoader snippets as executable C++ without an actual adaptation.

**Verified from official documentation — Autograd:** Python PyTorch accumulates gradients for leaves across backward calls, and version tracking detects many in-place modifications that would invalidate saved values. Numerical correctness must be checked in addition to whether backward runs without error.

**Verified from official documentation — CUDA memory:** Cached but unused allocations may remain visible to `nvidia-smi`; `empty_cache()` cannot free still-live tensors. A memory snapshot or detailed statistics are needed before concluding that a particular failure is fragmentation.

**Verified from official documentation — Transfers:** `non_blocking` can reduce CPU waiting even with pageable memory; meaningful transfer/compute overlap needs suitable streams, pinned buffers, independent work, and dependencies. Asynchronous GPU-to-CPU results require completion before host access, and pinned H2D sources must remain unchanged during pending transfers.

**Verified from official documentation — AMP:** Unscale before gradient clipping; GradScaler may skip invalid updates. Autocast does not make every operation use one dtype, and its details vary by device and release.

**Verified from official documentation — AdamW:** Weight decay is decoupled from adaptive momentum and variance estimation.

---

## 10. Practice: Building Micrograd and a Memory Calculator in C++

*Source: PDF pp. 107–108, with a C++ adaptation.*

The **original book exercise** asks learners to implement Python `Value.__add__`, `Value.__mul__`, `Value.tanh()`, a topologically ordered `backward()`, and `train_mem_bytes(n_params)`, validated by **seven repository pytest cases**. Those original starter files and tests are not part of the PDF itself. The following is a **new C++17 adaptation** of the same learning goals; it does not claim to be the original Python grader or to pass its Python tests.

### 10.1. Full C++17 scalar reverse-mode autodiff example

This code is self-contained, needs no PyTorch installation, and demonstrates (a) recorded dependencies, (b) local backward rules, (c) `+=` for multiple paths, (d) topological ordering, and (e) 16 bytes/parameter under the book's specified training-state assumptions. It is also suitable for ordinary C++ compilation as a standalone exercise.

```cpp
#include <cmath>
#include <cstdint>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <unordered_set>
#include <vector>

struct Value : std::enable_shared_from_this<Value> {
    double data = 0.0;
    double grad = 0.0;
    std::vector<std::shared_ptr<Value>> parents;
    std::function<void()> local_backward = [] {};

    explicit Value(double x) : data(x) {}

    void backward() {
        std::vector<std::shared_ptr<Value>> topo;
        std::unordered_set<const Value*> visited;
        auto dfs = [&](auto&& self, const std::shared_ptr<Value>& node) -> void {
            if (!visited.insert(node.get()).second) return;
            for (const auto& parent : node->parents) self(self, parent);
            topo.push_back(node);
        };
        dfs(dfs, shared_from_this());
        grad = 1.0; // Derivative of final output with respect to itself.
        for (auto it = topo.rbegin(); it != topo.rend(); ++it)
            (*it)->local_backward();
    }
};

using V = std::shared_ptr<Value>;
V make_value(double x) { return std::make_shared<Value>(x); }

V add(const V& a, const V& b) {
    auto out = make_value(a->data + b->data);
    out->parents = {a, b};
    std::weak_ptr<Value> weak_out = out; // Avoid a self-ownership cycle.
    out->local_backward = [a, b, weak_out] {
        auto o = weak_out.lock();
        if (!o) return;
        a->grad += o->grad;
        b->grad += o->grad;
    };
    return out;
}

V mul(const V& a, const V& b) {
    auto out = make_value(a->data * b->data);
    out->parents = {a, b};
    std::weak_ptr<Value> weak_out = out;
    out->local_backward = [a, b, weak_out] {
        auto o = weak_out.lock();
        if (!o) return;
        a->grad += b->data * o->grad;
        b->grad += a->data * o->grad;
    };
    return out;
}

V tanh_value(const V& a) {
    auto out = make_value(std::tanh(a->data));
    out->parents = {a};
    std::weak_ptr<Value> weak_out = out;
    out->local_backward = [a, weak_out] {
        auto o = weak_out.lock();
        if (!o) return;
        a->grad += (1.0 - o->data * o->data) * o->grad;
    };
    return out;
}

std::int64_t train_mem_bytes(std::int64_t n_params) {
    // BF16 parameter 2 + BF16 gradient 2 + FP32 master 4
    // + FP32 Adam first moment 4 + FP32 second moment 4 = 16.
    if (n_params < 0 || n_params > std::numeric_limits<std::int64_t>::max() / 16)
        throw std::invalid_argument("Invalid parameter count");
    return n_params * 16;
}

int main() {
    auto x = make_value(2.0);
    auto y = make_value(3.0);
    auto z = add(mul(x, y), x); // z = x*y+x = 8
    z->backward();

    std::cout << "z=" << z->data << "\n";       // 8
    std::cout << "dz/dx=" << x->grad << "\n";   // 4
    std::cout << "dz/dy=" << y->grad << "\n";   // 2
    std::cout << "1B train bytes=" << train_mem_bytes(1'000'000'000LL) << "\n";
    std::cout << "7B train bytes=" << train_mem_bytes(7'000'000'000LL) << "\n";
}
```

**Why topological order matters:** Shared intermediate nodes may be referenced multiple times, but their `local_backward` function should normally execute **once after all downstream contributions are accumulated**. The `visited` set prevents double execution, while `+=` preserves both paths in `x*y+x`.

**Important limitation of this teaching implementation:** It does not implement full PyTorch autograd safeguards for in-place mutations, full graph resets across repeated backwards, broadcasting, device execution, thread safety, or tensor-valued Jacobian rules. It is intended for one straightforward backward evaluation over an immutable scalar computation graph.

### 10.2. Worked exercise checks

| Exercise                                    | Expected result                                       |
| ------------------------------------------- | ----------------------------------------------------- |
| `z=x*y+x` with `x=2,y=3`                    | `z=8`, `dz/dx=4`, `dz/dy=2`                           |
| `tanh(x)` derivative                        | `1-tanh(x)^2` times upstream gradient                 |
| One billion parameters under 16 B/parameter | **16,000,000,000 bytes**                              |
| Seven billion parameters                    | **112,000,000,000 bytes = 112 GB**                    |
| Thirteen billion parameters                 | **208,000,000,000 bytes = 208 GB**                    |
| One value reused in multiple paths          | Sum all derivatives; do not replace previous gradient |
| Reused intermediate node                    | Execute local backward once per topological traversal |

To validate a scalar autodiff implementation independently, compare computed gradients against central finite differences with a moderate epsilon, accounting for rounding error. The original book's tests compare several fixed random inputs with numerical derivatives and use tolerances appropriate for a small CPU exercise.

---

## 11. Exam-Style Questions and Answers

### Tensor storage and broadcasting

**Q1.** A `(3,4)` tensor has strides `(4,1)` and offset `0`. What is the storage index of `[2,1]`?  
**A:** $2\cdot4+1=9$.

**Q2.** Why can transpose share data while `view(-1)` sometimes fails?  
**A:** Transpose changes dimension metadata; flattening in new logical order may not be expressible with a single constant stride. `reshape` may copy to make the target possible.

**Q3.** Is `contiguous()` guaranteed to allocate new memory?  
**A:** No; it may return the existing tensor if already contiguous in the requested format. Use `clone()` when a separate data copy is required.

**Q4.** Broadcast `(8,1,6,1)` with `(7,1,5)`.  
**A:** `(8,7,6,5)`.

**Q5.** Why can `(B,) - (B,1)` be a dangerous regression-loss expression?  
**A:** It produces pairwise differences of shape `(B,B)`, not aligned errors of shape `(B,)`.

**Q6.** If a `(1,M)` bias feeds `(N,M)` outputs, how is its gradient computed?  
**A:** Sum upstream gradients along the broadcasted batch axis, obtaining `(1,M)`.

**Q7.** How large is an FP32 `(1024,1024,1024)` intermediate?  
**A:** $1024^3\times4\,\mathrm{B}=4\,\mathrm{GiB}$.

### Autograd and module state

**Q8.** For $z=xy+x$ at $x=2,y=3$, what is $dz/dx$?  
**A:** $y+1=4$.

**Q9.** What is the difference between `grad_fn` and `.grad`?  
**A:** `grad_fn` identifies a backward operation/graph relationship; `.grad` stores accumulated derivative values (typically on trainable leaf parameters).

**Q10.** Why save forward activations?  
**A:** Backward formulas may require input values, outputs, masks, statistics, or other saved information. Retaining them can dominate memory use.

**Q11.** Why use reverse-mode autodiff for a scalar training loss with millions of parameters?  
**A:** It propagates a vector–Jacobian product backward through one graph rather than requiring finite differences for each parameter.

**Q12.** Why can an in-place update trigger a backward version-counter error?  
**A:** A value required by a saved backward rule has changed since forward; using it might produce an incorrect derivative.

**Q13.** A Python model runs layers stored in a plain list, but the optimizer misses them. Why?  
**A:** The parent module may not have registered those child modules. Use the appropriate module container/registration system.

**Q14.** Why does BatchNorm use buffers for `running_mean`/`running_var`?  
**A:** They are model state to save and move with the model, but are updated by forward statistics rather than ordinary weight gradients.

**Q15.** Why is `load_state_dict(..., strict=False)` risky after a DDP checkpoint key mismatch?  
**A:** It may ignore many/all mismatched keys without mapping `module.` to the expected names; inspect missing and unexpected keys.

### Training loops and optimizers

**Q16.** Why is `loss.backward(); optimizer.step()` without gradient clearing dangerous across independent batches?  
**A:** `.grad` accumulates previous batches' gradients; later updates no longer correspond to the intended current batch.

**Q17.** Four equal-sized micro-batches have mean gradients `2,4,6,8`. What should the accumulated mean be?  
**A:** `5`, achieved by scaling each mean loss by `1/4` before backward.

**Q18.** Is a missing gradient the same as a zero gradient under momentum SGD?  
**A:** No. A missing gradient commonly skips the parameter, while an explicit zero gradient can still apply momentum/decay.

**Q19.** For Adam with $g_1=2$, zero moments, $\beta_1=0.9,\beta_2=0.999$, what are first corrected moments?  
**A:** $\hat m_1=2$, $\hat v_1=4$.

**Q20.** What distinguishes AdamW from adding L2 decay to Adam's gradient?  
**A:** AdamW decouples parameter shrinkage from the data-gradient moment estimates.

**Q21.** A 10-epoch run has 100 batches per epoch. What happens if a schedule intended for epoch updates is advanced every batch?  
**A:** It gets 1000 updates instead of 10; its intended time unit is wrong.

**Q22.** Global norm clipping of `[3,4]` to `1` gives what gradient?  
**A:** Approximately `[0.6,0.8]`, preserving direction.

**Q23.** Does `model.eval()` remove the autograd graph?  
**A:** No. It changes training-sensitive module behavior, while no-grad/inference mode controls gradient recording.

### Precision and GPU memory

**Q24.** What are FP16's largest finite value and smallest positive normal/subnormal values?  
**A:** `65504`, about `6.10e-5` normal, and `5.96e-8` subnormal.

**Q25.** Why can BF16 train more robustly than FP16 despite fewer mantissa bits?  
**A:** Its much wider exponent range can avoid some FP16 overflow/underflow; BF16 still has rounding and precision limitations.

**Q26.** Why doesn't a loss-scaling operation after a gradient has already rounded to zero fix underflow?  
**A:** Lost information cannot be recovered; the scale must affect backward before that rounding occurs.

**Q27.** Why does a BF16 computation path often retain FP32 parameters for updates?  
**A:** To accumulate updates smaller than the low-precision representable spacing at the parameter's magnitude.

**Q28.** Does TF32 reduce FP32 tensor storage to 2 bytes per value?  
**A:** No. It changes eligible internal compute precision while FP32 tensor storage remains 4 bytes.

**Q29.** Estimate static full-training memory for 7B parameters using 16 B/parameter.  
**A:** **112 GB**, excluding activations and other overhead.

**Q30.** Why is the first `optimizer.step()` important when measuring VRAM?  
**A:** Adam's `m`/`v` may be lazily created at that point.

**Q31.** Why does `memory_reserved()` often exceed `memory_allocated()`?  
**A:** PyTorch's caching allocator retains unused blocks for reuse; the gap alone does not establish a leak.

**Q32.** Can reducing batch from 4 to 1 fix an OOM caused by static 112 GB parameter state on an 80 GiB device?  
**A:** Not by itself. The static state exceeds capacity before activations are considered.

**Q33.** Why does a naïve Transformer activation model contain an $s^2$ term?  
**A:** It can explicitly store attention-related values over all pairs of sequence positions.

### CUDA execution and data loading

**Q34.** Why can a Python/C++ host-side timer underestimate a CUDA matrix multiplication?  
**A:** The call may only enqueue GPU work; the timer must include completion or use appropriate device events.

**Q35.** Does `non_blocking` automatically overlap copying the current batch with computing on that batch?  
**A:** No. The current batch's computation depends on its copy completing. Overlap is often with the **next** batch on another stream.

**Q36.** Why can manually pinning each CPU batch slow down the pipeline?  
**A:** The page-locking/copy preparation has a cost that may outweigh its subsequent H2D savings.

**Q37.** One worker needs `40 ms/batch`, and the GPU needs `12 ms/batch`. What worker count does the ideal supply calculation suggest?  
**A:** At least $40/12\approx3.33$, hence about **four** workers as a test candidate.

**Q38.** Can a larger `prefetch_factor` solve a permanent average throughput deficit?  
**A:** No. Buffering absorbs temporary fluctuations but cannot raise long-run worker supply rate above its capacity.

**Q39.** Why can a GPU show 100% utilization despite poor useful model throughput?  
**A:** The utilization statistic measures time with active kernels, not the fraction of theoretical compute throughput delivered for model FLOPs.

**Q40.** How would you prove a transfer or DataLoader optimization helps?  
**A:** Hold workload constant, synchronize appropriately, compare end-to-end throughput and overlap timelines, repeat trials, validate correctness, and include pinning/IPC/host-memory overhead.

---

## 12. One-Page Formula and Debugging Cheat Sheet

| Topic                                        | Formula or quick check                                       |
| -------------------------------------------- | ------------------------------------------------------------ |
| Tensor storage index                         | $\mathrm{offset}+\sum_k i_k\,s_k$                            |
| Broadcast compatibility                      | Right-align; sizes equal or one equals 1                     |
| Broadcasted bias gradient                    | $\partial L/\partial b_j=\sum_i\partial L/\partial y_{ij}$   |
| FP32 tensor storage                          | $\mathrm{numel}\times4$ bytes                                |
| BF16/FP16 tensor storage                     | $\mathrm{numel}\times2$ bytes                                |
| Linear layer parameter count                 | $\mathrm{in}\times\mathrm{out}+\mathrm{out}$ (with bias)     |
| Simple dense training FLOPs                  | $\approx6ND$                                                 |
| Numerical gradient check                     | $[L(\theta+\varepsilon)-L(\theta-\varepsilon)]/(2\varepsilon)$ |
| Adam's moments                               | $m_t=\beta_1m_{t-1}+(1-\beta_1)g_t$; $v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2$ |
| AdamW decoupled shrinkage                    | $\theta\leftarrow(1-\eta\lambda)\theta$ plus adaptive data-gradient update |
| Global norm clipping                         | $g\leftarrow g\min(1,c/\lVert g\rVert_2)$                    |
| Training parameter-state estimate            | $N\times\mathrm{bytes/parameter}$                            |
| Book's naïve Transformer activation estimate | $sbhL(34+5as/h)$ bytes, **specific assumptions**             |
| Layer-boundary-only recomputation estimate   | $2sbhL$ bytes, **not a peak**                                |
| Ideal worker count lower-bound heuristic     | $W\gtrsim t_{\mathrm{CPU}}/t_{\mathrm{GPU}}$                 |
| GPU-Util                                     | Fraction of sampled time with active GPU kernel work         |
| MFU                                          | Estimated useful model FLOPs per second divided by matching hardware peak FLOPS |

**First diagnose, then optimize:**

- **Unexpected output/loss:** Check tensor shapes and label alignment **before** investigating kernels.
- **Missing parameter updates:** Check registration → requires-grad → `.grad` presence → optimizer membership → scheduler.
- **Incorrect gradients:** Reduce to a scalar example; check multiple paths, reduction shapes, saved values, and in-place mutations.
- **First-step OOM:** Separate weights, gradients, newly created optimizer state, activations, and temporary allocations.
- **High GPU-Util with slow throughput:** Profile useful FLOPs, kernel efficiency, bandwidth, and host synchronization.
- **Poor data pipeline throughput:** Compare a preloaded GPU batch, CPU worker supply, H2D transfer, pinning cost, and prefetching.
- **Suspicious speedups:** Warm up, include completion, use consistent timing boundaries, repeat, and check correctness.

