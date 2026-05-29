<div align="center">

<img src="https://img.shields.io/badge/ICML-2026-blue?style=flat-square&logo=academia" />
<img src="https://img.shields.io/badge/Python-3.8+-3776AB?style=flat-square&logo=python&logoColor=white" />
<img src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" />
<img src="https://img.shields.io/badge/License-MIT-green?style=flat-square" />

<br/><br/>

# 🌉 TESS

### **From Text to Forecasts: Bridging Modality Gap with Temporal Evolution Semantic Space**

<br/>

*Lehui Li\* · Yuyao Wang\* · Jisheng Yan\* · Wei Zhang · Jinliang Deng† · Haoliang Sun · Zhongyi Han · Yongshun Gong†*


<br/>

[![Paper](https://img.shields.io/badge/📄_Paper-ICML_2026-blue?style=for-the-badge)](https://arxiv.org)
[![Code](https://img.shields.io/badge/💻_Code-GitHub-black?style=for-the-badge)](https://github.com/olivia3395/TESS)
[![Poster](https://img.shields.io/badge/🖼️_Poster-PDF-orange?style=for-the-badge)](#)

</div>


## Overview

Real-world time series are often disrupted by external events — news, weather shocks, market sentiment — that cause abrupt, unpredictable shifts. While text naturally describes these events, existing multimodal methods fail to translate qualitative language into reliable numerical forecasting signals. **TESS** introduces a **Temporal Evolution Semantic Space** as an intermediate bottleneck: instead of feeding raw text directly to a forecaster, an LLM distills it into compact, interpretable temporal primitives, which then condition the forecasting model as structured exogenous signals.

<br/>

<div align="center">

| Challenge | TESS's Solution |
|:---|:---|
| Models over-attend to redundant text tokens | Distill text into compact temporal primitives |
| Qualitative language resists numerical decoding | Define numerically grounded primitives (shift, volatility, shape, lag) |
| LLM extraction can be noisy | Confidence-aware gating suppresses unreliable primitives |
| Direct fusion leads to unstable training | Semantic prefix tokens enable smooth, fast convergence |

</div>



##  Method: Three-Stage Pipeline

<div align="center">
<img src="assets/tess_overview.png" width="90%" alt="TESS Overview"/>
<br/><sub><b>Figure 1:</b> Overview of TESS. Text is distilled into temporal primitives by a frozen LLM, filtered by confidence-aware gating, and injected as semantic prefix tokens into a Transformer forecaster.</sub>
</div>

<br/>

### Stage 1 · Temporal Evolution Semantic Space

We define four **Temporal Semantic Primitives (TSPs)** — each verifiable from the actual forecast window, providing reliable training supervision:

<div align="center">

| Challenge | TESS's Solution |
|:---|:---|
| Models over-attend to redundant text tokens | Distill text into compact temporal primitives |
| Qualitative language resists numerical decoding | Define numerically grounded primitives (shift, volatility, shape, lag) |
| LLM extraction can be noisy | Confidence-aware gating suppresses unreliable primitives |
| Direct fusion leads to unstable training | Semantic prefix tokens enable smooth, fast convergence |

</div>


### Stage 2 · Text → Temporal Semantic Primitives

A frozen LLM classifies each primitive from input text under a structured prompt. A **confidence-aware gating network** then estimates extraction reliability using the log-probability margin between the top-1 and top-2 candidates — suppressing noisy primitives during inference via soft weighting.

### Stage 3 · Semantic Primitives-Conditioned Forecasting

Gated primitive embeddings are prepended as **semantic prefix tokens** to the patch embeddings of a PatchTST backbone. This allows semantic signals to participate in all Transformer attention layers. The model trains end-to-end (LLM frozen) with a joint forecasting + gating supervision loss.



##  Quick Start

### Installation

### Running Experiments



##  Key Hyperparameters

| Parameter | Description | Default |
|:---|:---|:---:|
| `--lambda_gate` | Weight for gating supervision loss | `0.1` |
| `--temperature` | LLM softmax temperature for primitive extraction | `1.0` |
| `--patch_len` | Patch length | `16` |
| `--stride` | Patch stride | `8` |
| `--d_model` | Transformer hidden dimension | `128` |
| `--n_heads` | Number of attention heads | `8` |
| `--n_layers` | Number of Transformer encoder layers | `3` |
| `--epoch` | Training epochs | `100` |
| `--patience` | Early stopping patience | `10` |
| `--lr` | Learning rate (AdamW) | `1e-4` |
| `--llm_model` | LLM for primitive extraction | `Qwen3-8B` |

>  **Tip:** Use `--use_cv True` for automatic hyperparameter selection via cross-validation.




