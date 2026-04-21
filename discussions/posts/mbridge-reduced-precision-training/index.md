---
date:
  created: 2025-11-18
slug: mbridge-reduced-precision-training
authors:
  - aditya_vavre
  - nima_tajbakhsh
categories:
    - Low-Precision Training
    - Megatron-Bridge
tags:
    - Low-Precision Training
    - Megatron-Bridge
---

# Low-Precision Training with Megatron-Bridge: FP8 (Current Scaling), MXFP8, NVFP4 vs BF16

<!--
nemo_discussion: {
  "repo": "https://github.com/NVIDIA-NeMo/Megatron-Bridge",
  "authors": ["aditya_vavre", "nima_tajbakhsh"]
}
-->

## Introduction

In the ever-evolving field of deep learning, efficient training is crucial to handle larger models and datasets within practical resource constraints. Low-precision training has emerged as a powerful approach, enabling faster training and significant GPU memory savings without compromising model accuracy.

This post delves into recent advances in low-precision training, specifically focusing on FP8 per-tensor current scaling (FP8-CS), MXFP8, and NVFP4 precisions. We will compare these to the more established BF16 precision training and discuss the practical advantages of adopting these newer formats.

## What is Low-Precision Training?

Low-precision training involves using numerical formats with fewer bits to represent weights, activations, and gradients during neural network training. This reduces memory bandwidth and computational demand, leading higher training throughput. NVIDIA Blackwell GPUs provide native support for general matrix multiplications (GEMMs) for a wide range of microscaling formats: MXFP8, MXFP6, MXFP4, NVFP4.
<!-- more -->

## Overview of Precision Formats

<a name="figure1"></a>

<div style="display: flex; justify-content: space-around; align-items: center;">
  <img src="assets/precision_comparison.png" alt="Comparison of FP8, MXFP8 and NVFP4 precision formats">
</div>

<p style="text-align: center;"><strong>Figure 1:</strong> Comparison of FP8, MXFP8 and NVFP4 precision formats.</p>

### [FP8-CS](https://developer.nvidia.com/blog/floating-point-8-an-introduction-to-efficient-lower-precision-ai-training/)

FP8-CS applies FP8 to linear layers via scaling factors based on the statistical properties of each tensor in the **current** training iteration. It is therefore more reactive to the immediate dynamic range of the tensor, potentially leading to more accurate quantization in each step. FP8-CS uses the E4M3 format for representing weights and activations, and the E5M2 format to represent gradients (due to its wide dynamic range).

### [MXFP8](https://developer.nvidia.com/blog/floating-point-8-an-introduction-to-efficient-lower-precision-ai-training/)

MXFP8 is an exemplification of block scaling for the NVIDIA Blackwell architecture. With MXFP8, each contiguous block of 32 values in a tensor is assigned its own scaling factor, computed directly by the GPU’s Tensor Cores. All values are encoded in E4M3 datatype while the scaling factors themselves use the E8M0 datatype. This finer-grained scaling mechanism can better capture the data distribution during training and result in better model convergence.

### [NVFP4](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/)

NVFP4 further improves memory efficiency and throughput by using the E2M1 4-bit format. It maintains numerical accuracy across a wide dynamic range of tensor values by utilizing (1) a E4M3 scaling factor to each 16-value micro-block (2) a second-level FP32 scalar applied per tensor.

## Empirical Comparison: Llama3-8B and Research-8B Models

To demonstrate the practical impact of low-precision training, we evaluate the convergence and downstream task performance of low-precision pretraining across two dense transformer model architectures, Llama3-8B and an NVIDIA internal research 8B model (Research-8B with dense grouped query attention (GQA) architecture that is similar to Llama 3 8B), trained on up to 1 trillion tokens. Our goal is to assess whether emerging low-precision formats can match the convergence behavior and final model quality of conventional BF16 training.

### Experimental Setup

We ran a set of large-scale pretraining experiments using four different numeric precisions: BF16 (default baseline), FP8-CS, MXFP8 and NVFP4. 

Note that MXFP8 and NVFP4 are natively supported ONLY on NVIDIA Blackwell GPUs. We trained Llama3-8B and Research-8B under identical settings (optimizer, learning rate schedule, etc.) so we could directly compare the impact of precision. Training was conducted using [NeMo Megatron-Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge) on NVIDIA B200 GPUs with a batch size of 768, maximum learning rate of 6e-4, cosine decay to 6e-6, and a warmup of approximately 8.4B tokens.

For FP8-CS, we keep the first and last transformer layers in BF16 precision to maintain numerical stability. For NVFP4, following the setup and recipe from [Pretraining Large Language Models with NVFP4](https://arxiv.org/abs/2509.25149), the final 4 transformer layers (<15% of total layers) are retained in BF16. We denote this as F0L4.

We experiment with two datasets: 1. [Lingua DCLM Dataset](https://github.com/facebookresearch/lingua) which is a high-quality curated pretraining dataset sourced from Common Crawl. 2. Internal research dataset which is a blend of [NVIDIA's released pretraining corpora](https://huggingface.co/collections/nvidia/nemotron-pre-training-dataset). We trained Llama3-8B on both datasets and Research-8B on the internal research dataset.

### Convergence Behavior

Let's start with training dynamics. The figures below show training and validation loss curves for all four precisions.

<a name="figure2"></a>

<div style="display: flex; justify-content: space-around; align-items: center;">
  <img src="assets/fig1_llama3_dclm.png" alt="Train and validation loss curves for Llama3-8B (DCLM dataset)">
</div>

<p style="text-align: center;"><strong>Figure 2:</strong> Training and validation loss curves for Llama3-8B model trained on the DCLM dataset. Training stability is maintained across precisions. Although NVFP4 shows slightly higher loss, downstream accuracies remain unaffected (see Table 1).</p>

<a name="figure3"></a>

<div style="display: flex; justify-content: space-around; align-items: center;">
  <img src="assets/fig2_llama3_research_data.png" alt="Train and validation loss curves for Llama3-8B (research dataset)">
</div>

<p style="text-align: center;"><strong>Figure 3:</strong> Training and validation loss curves for Llama3-8B model trained on the internal research dataset. Although NVFP4 shows slightly higher loss, downstream accuracies remain unaffected (see Table 1)</p>

<a name="figure4"></a>

<div style="display: flex; justify-content: space-around; align-items: center;">
  <img src="assets/fig3_research_8b_research_data.png" alt="Train and validation loss curves for Research-8B (research dataset)">
</div>

<p style="text-align: center;"><strong>Figure 4:</strong> Training and validation loss curves for Research-8B model trained on the internal research dataset. Although NVFP4 shows slightly higher loss, downstream accuracies remain unaffected (see Table 1)</p>

Across both models and datasets, the low-precision runs closely track the BF16 baseline in both training and validation loss. FP8-CS and MXFP8 almost perfectly overlap with BF16 throughout the entire training process, showing no signs of instability or divergence. NVFP4, despite operating at a much lower 4-bit precision, demonstrates stable convergence and achieves a final validation loss within ~1% of BF16. Despite the small gap in loss, downstream task accuracies remain largely unaffected as we show below.

### Downstream Evaluation

We also measured how these models perform after pretraining on standard downstream benchmarks: MMLU (5-shot), Winogrande (5-shot), HellaSwag (10-shot; norm accuracy), and ARC-Challenge (25-shot; norm accuracy). All evaluations were run in BF16 precision to isolate the impact of training precision. We report the performance of the last checkpoint.

The results are shown in [Table 1](#table1) and tell a consistent story: models trained with low precision achieve downstream accuracies nearly identical to the BF16 baseline. MXFP8 slightly outperforms FP8-CS, likely due to fine-grained scaling design offering better dynamic range within a tensor. NVFP4 maintains downstream accuracy within <1% of BF16 across all benchmarks.

<a name="table1"></a>

<div style="display: flex; justify-content: center;">
<table>
  <thead>
    <tr>
      <th>Model</th>
      <th>Dataset</th>
      <th>Precision</th>
      <th style="text-align: center;">MMLU (↑)</th>
      <th style="text-align: center;">Hellaswag (↑)</th>
      <th style="text-align: center;">Winogrande (↑)</th>
      <th style="text-align: center;">ARC-C (↑)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="4">Llama3-8B</td>
      <td rowspan="4">DCLM</td>
      <td>BF16</td>
      <td style="text-align: center;">45.98</td>
      <td style="text-align: center;">76.44</td>
      <td style="text-align: center;">70.17</td>
      <td style="text-align: center;">51.28</td>
    </tr>
    <tr>
      <td>FP8-CS</td>
      <td style="text-align: center;">46</td>
      <td style="text-align: center;">75.25</td>
      <td style="text-align: center;">70.24</td>
      <td style="text-align: center;">49.91</td>
    </tr>
    <tr>
      <td>MXFP8</td>
      <td style="text-align: center;">46.56</td>
      <td style="text-align: center;">75.46</td>
      <td style="text-align: center;">71.27</td>
      <td style="text-align: center;">51.11</td>
    </tr>
    <tr>
      <td>NVFP4</td>
      <td style="text-align: center;">45.64</td>
      <td style="text-align: center;">75.59</td>
      <td style="text-align: center;">69.38</td>
      <td style="text-align: center;">51.28</td>
    </tr>
    <tr>
      <td rowspan="4">Llama3-8B</td>
      <td rowspan="4">Research dataset</td>
      <td>BF16</td>
      <td style="text-align: center;">52.73</td>
      <td style="text-align: center;">75.71</td>
      <td style="text-align: center;">67.88</td>
      <td style="text-align: center;">51.37</td>
    </tr>
    <tr>
      <td>FP8-CS</td>
      <td style="text-align: center;">52.46</td>
      <td style="text-align: center;">75.65</td>
      <td style="text-align: center;">70.17</td>
      <td style="text-align: center;">54.52</td>
    </tr>
    <tr>
      <td>MXFP8</td>
      <td style="text-align: center;">53.7</td>
      <td style="text-align: center;">75.54</td>
      <td style="text-align: center;">69.69</td>
      <td style="text-align: center;">51.62</td>
    </tr>
    <tr>
      <td>NVFP4</td>
      <td style="text-align: center;">52.83</td>
      <td style="text-align: center;">75.04</td>
      <td style="text-align: center;">71.98</td>
      <td style="text-align: center;">53.58</td>
    </tr>
    <tr>
      <td rowspan="4">Research-8B</td>
      <td rowspan="4">Research dataset</td>
      <td>BF16</td>
      <td style="text-align: center;">53</td>
      <td style="text-align: center;">76.98</td>
      <td style="text-align: center;">70.4</td>
      <td style="text-align: center;">55.89</td>
    </tr>
    <tr>
      <td>FP8-CS</td>
      <td style="text-align: center;">52.62</td>
      <td style="text-align: center;">75.81</td>
      <td style="text-align: center;">70.8</td>
      <td style="text-align: center;">54.44</td>
    </tr>
    <tr>
      <td>MXFP8</td>
      <td style="text-align: center;">52.38</td>
      <td style="text-align: center;">76.55</td>
      <td style="text-align: center;">69.77</td>
      <td style="text-align: center;">53.58</td>
    </tr>
    <tr>
      <td>NVFP4</td>
      <td style="text-align: center;">52.21</td>
      <td style="text-align: center;">76.19</td>
      <td style="text-align: center;">70.32</td>
      <td style="text-align: center;">54.95</td>
    </tr>
  </tbody>
</table>
</div>

<p style="text-align: center;"><strong>Table 1:</strong> Downstream task accuracy (in %) for Llama3-8B and Research-8B models. Despite small differences in training/validation loss, all precision formats achieve similar downstream task accuracy.</p>

### Task Performance vs. Tokens Consumed

To further analyze convergence in terms of knowledge acquisition, we plot MMLU accuracy against tokens consumed during pretraining in [Figure 5](#figure5). The curves show that all low-precision formats exhibit nearly the same learning trajectories to BF16 during pretraining. NVFP4 initially lags slightly during early training (<200B tokens) on the DCLM dataset, but catches up and converges to the same plateau by ~700B tokens. This demonstrates that NVFP4 preserves learning efficiency over long-horizon pretraining.

<a name="figure5"></a>

<div style="text-align: center;">
  <img src="assets/fig4_mmlu_vs_tokens.png" alt="MMLU accuracy vs tokens consumed during pretraining">
</div>

<p style="text-align: center;"><strong>Figure 5:</strong> MMLU accuracy vs tokens consumed during pretraining.</p>

### Ablation Study: How Many Layers Need BF16 for Stable NVFP4 Training?
Although NVFP4 enables highly efficient 4-bit training, our experiments show that some layers must remain in BF16 precision to ensure stable optimization and strong downstream performance. To understand this behavior, we conducted an ablation over different FxLy configurations (where x is the number of initial layers kept in BF16, and y is the number of final layers kept in BF16).

We evaluated the following configurations: F0L0, F0L1, F0L2, F0L4, F1L1, and compared both their training dynamics ([Figure 6](#figure6)) and MMLU performance ([Figure 7](#figure7)).

#### Stability and Convergence

As shown in [Figure 6](#figure6), F0L0 (fully NVFP4) diverges early, indicating that NVFP4 requires keeping some layers in BF16 for stable training. All other configurations (F0L1, F0L2, F0L4, and F1L1) train stably, with smooth training and validation loss curves.

#### Effect of BF16 Layer Placement (F0L2 vs. F1L1)

To isolate whether BF16 matters more at the start or end of the network, we compare F0L2 (two BF16 layers at the end) with F1L1 (one at the beginning, one at the end). Both use two BF16 layers total. F1L1 shows higher training loss (1.84081 vs. 1.83577, +0.27%), higher validation loss (1.35356 vs. 1.35003, +0.26%), and slightly lower downstream MMLU accuracy throughout training (as shown in [Figure 7](#figure7)). This consistent gap shows that BF16 layers near the end of the network have a greater impact on final model quality.

#### Effect of More BF16 Layers

Across all configurations, adding more BF16 layers improves convergence and downstream accuracy, but with diminishing returns. Keeping the last four layers in BF16 (F0L4) is sufficient to match the quality of heavier BF16 configurations while retaining the efficiency benefits of NVFP4.

<a name="figure6"></a>

<div style="display: flex; justify-content: space-around; align-items: center;">
  <img src="assets/fig5_fxly.png" alt="Train and validation loss curves for Research-8B (research dataset)">
</div>

<p style="text-align: center;"><strong>Figure 6:</strong> Training and validation loss curves for Research-8B model trained on the internal research dataset for different FxLy configurations.</p>

<a name="figure7"></a>

<div style="text-align: center;">
  <img src="assets/fxly_mmlu_vs_tokens.png" alt="MMLU accuracy vs tokens consumed during pretraining"  style="width: 40%;">
</div>

<p style="text-align: center;"><strong>Figure 7:</strong> MMLU accuracy vs tokens consumed during pretraining for different FxLy configurations. FP8-CS shown as a baseline.</p>

### Ablation Study: Does Training Recipe Impact NVFP4 Convergence?
We ran a focused recipe ablation on Llama3-8B with the DCLM dataset using three pretraining configurations (see [Table 2](#table2) below) to test how optimizer hyperparameters and batch/LR schedules interact with NVFP4 low-precision format. 

<a name="table2"></a>

<div style="display: flex; justify-content: center;">
<table>
  <thead>
    <tr>
      <th>Recipe Config (Learning Rate Max → Min | Global Batch Size | AdamW ε)</th>
      <th style="text-align: center;">BF16</th>
      <th style="text-align: center;">NVFP4 F0L4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Recipe0: LR 3e-4 → 3e-5 | GBS 512 | AdamW ε = 1e-5</td>
      <td style="text-align: center;">44.06</td>
      <td style="text-align: center;">27.48</td>
    </tr>
    <tr>
      <td>Recipe1: LR 6e-4 → 6e-6 | GBS 768 | AdamW ε = 1e-8</td>
      <td style="text-align: center;">47.30</td>
      <td style="text-align: center;">46.10</td>
    </tr>
    <tr>
      <td>Recipe2: LR 6e-4 → 6e-6 | GBS 768 | AdamW ε = 1e-5</td>
      <td style="text-align: center;">45.71</td>
      <td style="text-align: center;">43.63</td>
    </tr>
  </tbody>
</table>
</div>

<p style="text-align: center;"><strong>Table 2:</strong> MMLU (5-shot) accuracy, reported as the maximum of the final three checkpoints for BF16 and NVFP4 across all recipe configurations.</p>

<a name="figure8"></a>

<div style="text-align: center;">
  <img src="assets/recipe_ablations.png" alt="Training loss across different recipe configurations for Llama3-8B on DCLM dataset."  style="width: 50%;">
</div>

<p style="text-align: center;"><strong>Figure 8:</strong>Training loss across different recipe configurations for Llama3-8B on DCLM dataset.</p>

[Figure 8](#figure8) shows that Recipe0 and Recipe2 converge significantly slower. The key issue is the AdamW $\epsilon$ term, which regularizes the denominator $\sqrt{v_t} + \epsilon$. When $\epsilon$ is large (1e-5), it dominates the denominator because $\sqrt{v_t}$ is very small due to gradient-norm clipping. This effectively removes second-moment adaptation:
- AdamW update: $w = w - \text{lr} \cdot \frac{m_t}{\sqrt{v_t} + \epsilon}$
- If $\epsilon \gg \sqrt{v_t}$, then $w \approx w - \frac{\text{lr}}{\epsilon} \cdot m_t$ (SGD with momentum)

Our experiments indicate that when using a $\epsilon$ value of 1e-5, the $\sqrt{v_t}$ term is approximately 1 order of magnitude smaller than $\epsilon$. Thus, a large $\epsilon$ collapses AdamW into SGD which means step magnitudes shrink (slow convergence) and loss of per-parameter normalization. SGD is much more sensitive to gradient noise because it no longer normalizes updates using the second moment; noisy gradients translate directly into noisy momentum, which in turn produces noisy weight updates (low signal to noise ratio). NVFP4 introduces additional quantization noise on top of the noise induced by large $\epsilon$. This amplified noise directly harms convergence and leads to degraded downstream accuracy. This is why Recipe0 performs poorly (small batch, large $\epsilon$, low LR → lowest SNR and the strongest SGD-like behavior). Recipe2, despite sharing the same $\epsilon$, performs better because its larger batch size reduces gradient noise, stabilizing updates and limiting NVFP4 drift, thus resulting in a smaller BF16–NVFP4 accuracy gap compared to Recipe0.

### Key Insights

- Low precision formats (FP8, MXFP8, NVFP4) achieve pretraining and validation losses very close to BF16, showing minimal degradation.
- Downstream task accuracy remains virtually unaffected, demonstrating that low-precision training maintains model effectiveness.
- MXFP8 often performs slightly better than standard FP8, due to its finer-grained scaling mechanism.
- NVFP4, despite its aggressive compression, still delivers competitive results when properly calibrated. The empirical sweet spot in our runs is Recipe1 (AdamW $\epsilon$=1e-8, LR=6e-4 → 6e-6, GBS=768) paired with NVFP4-F0L4: it retains NVFP4’s efficiency while matching BF16 quality.

## Advantages of Using FP8, MXFP8, NVFP4

<a name="table3"></a>

<div style="display: flex; justify-content: center;">
<table>
  <thead>
    <tr>
      <th>Precision</th>
      <th style="text-align: center;">Micro Batch Size</th>
      <th style="text-align: center;">Throughput (TFLOPS/GPU)</th>
      <th style="text-align: center;">Speedup vs BF16</th>
      <th style="text-align: center;">Step Time (s)</th>
      <th style="text-align: center;">Time to train 100B tokens (hr)</th>
      <th style="text-align: center;">Time Reduction</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>BF16</td>
      <td style="text-align: center;">2</td>
      <td style="text-align: center;">1165</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">5.79</td>
      <td style="text-align: center;">153</td>
      <td style="text-align: center;">-</td>
    </tr>
    <tr>
      <td>FP8-CS (F1L1)</td>
      <td style="text-align: center;">2</td>
      <td style="text-align: center;">1547</td>
      <td style="text-align: center;">1.33</td>
      <td style="text-align: center;">4.36</td>
      <td style="text-align: center;">115</td>
      <td style="text-align: center;">-25%</td>
    </tr>
    <tr>
      <td>MXFP8</td>
      <td style="text-align: center;">2</td>
      <td style="text-align: center;">1540</td>
      <td style="text-align: center;">1.32</td>
      <td style="text-align: center;">4.38</td>
      <td style="text-align: center;">116</td>
      <td style="text-align: center;">-24%</td>
    </tr>
    <tr>
      <td>NVFP4 (F0L4)</td>
      <td style="text-align: center;">4</td>
      <td style="text-align: center;">1850</td>
      <td style="text-align: center;">1.59</td>
      <td style="text-align: center;">3.64</td>
      <td style="text-align: center;">96</td>
      <td style="text-align: center;">-37%</td>
    </tr>
  </tbody>
</table>
</div>

<p style="text-align: center;"><strong>Table 3:</strong> Performance comparison for Llama3-8B training (GBS=128, Seq. Length=8192). FxLy denotes first 'x' layers and last 'y' transformer block layers are kept in BF16 precision.</p>

### 1. Faster End-to-End Training

Using 8-bit or 4-bit numeric formats drastically reduces computational overhead by allowing GPUs to process more operations per clock cycle. [Table 3](#table3) shows that, on NVIDIA Blackwell (GB200) GPUs, lower-precision formats deliver substantial throughput improvements and shorter step times compared to traditional BF16 training. Specifically, FP8-CS achieves a 1.33× throughput improvement, and NVFP4 format pushes efficiency further, delivering 1.59× higher throughput than BF16.



### 2. GPU Memory Savings

[Table 4](#table4) provides a detailed breakdown of memory usage across training components for different pretraining precisions. Using lower bit-width formats reduces the memory footprint of weights and activations, allowing larger models or batch sizes on the same hardware. As shown in [Table 3](#table3), NVFP4's efficiency enables the micro-batch size to double (from 2 to 4) during pretraining, directly improving throughput and scalability.

<a name="table4"></a>

<div style="display: flex; justify-content: center;">
<table>
  <thead>
    <tr>
      <th></th>
      <th></th>
      <th></th>
      <th colspan="3" style="text-align: center;">Optimizer</th>
      <th></th>
    </tr>
    <tr>
      <th>Precision</th>
      <th>Parameter</th>
      <th>Gradients</th>
      <th>Momentum</th>
      <th>Variance</th>
      <th>Master Parameter</th>
      <th>Others</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>FP16</td>
      <td>FP16</td>
      <td>FP32</td>
      <td rowspan="5">FP32</td>
      <td rowspan="5">FP32</td>
      <td rowspan="5">FP32</td>
      <td></td>
    </tr>
    <tr>
      <td>BF16</td>
      <td>BF16</td>
      <td>BF16</td>
      <td></td>
    </tr>
    <tr>
      <td>FP8 (tensor scaling)</td>
      <td>FP8x2</td>
      <td>BF16</td>
      <td>Scaling factor per weight tensor</td>
    </tr>
    <tr>
      <td>MXFP8</td>
      <td>FP8x2</td>
      <td>BF16</td>
      <td>(Scaling factor per 32 elements) x 2</td>
    </tr>
    <tr>
      <td>NVFP4</td>
      <td>FP4</td>
      <td>BF16</td>
      <td>16x16 2D block scales replicated for each 1x16 block</td>
    </tr>
  </tbody>
</table>
</div>

<p style="text-align: center;"><strong>Table 4:</strong> Memory footprint across training components for different precision formats.</p>

## Challenges and Considerations

- **Hardware Support:** MXFP8 and NVFP4 formats require the adoption of NVIDIA Blackwell GPUs.
- **BF16 Layer Requirements:** Ablation studies show that fully NVFP4 models (F0L0) diverge, and stable training requires keeping some layers in BF16. BF16 layers near the end of the network are especially important for mitigating NVFP4 quantization error.
- **Recipe Dependence:** Stable NVFP4 performance depends on using well-tuned training recipes (e.g., small AdamW $\epsilon$, appropriate LR schedule, and sufficiently large batch size). Poorly tuned recipes exacerbate noise and lead to significant quality degradation.

## Training with Low Precisions using [Megatron-Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge)

Megatron-Bridge enables users to train their models in low-precision. Low-precision recipes for Llama3-8B are available [here](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/5c7ebe7d8c31e75ddf882e1e6ced69d10e250786/src/megatron/bridge/recipes/llama/llama3.py#L240). You can use the recipes following the sample code below:
```python
from megatron.bridge.recipes.llama import llama3_8b_low_precision_pretrain_config as low_precision_pretrain_config
from megatron.bridge.training.gpt_step import forward_step

precision = "bf16_with_fp8_current_scaling_mixed"  # should be one of ["bf16_with_mxfp8_mixed", "bf16_with_fp8_current_scaling_mixed", "bf16_with_nvfp4_mixed"]
cfg = low_precision_pretrain_config(
    mixed_precision_recipe = precision,
    train_iters = 100,
    lr_warmup_iters = 10,
    lr_decay_iters = 90,
    mock = True,  # use mock dataset
)
pretrain(config=cfg, forward_step_func=forward_step)
```
Try out our simple python notebook which showcases how one can use the low-precision recipes to speed up their training workloads: [Quickstart Notebook](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/tutorials/training/reduced_precision_training.ipynb)

## Conclusion

Low-precision training formats like FP8 (current scaling), MXFP8, and NVFP4 offer exciting new avenues for faster, more efficient deep learning training compared to the widely adopted BF16. Their advantages in speed and memory savings open doors for training larger, more complex models. Empirical evidence from Llama3-8B and Research-8B models confirms that training with low precisions can match BF16 performance on both pretraining metrics and downstream tasks. As software frameworks evolve, we expect these formats to become mainstream, pushing the boundaries of AI capabilities.
