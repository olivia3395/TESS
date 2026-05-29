<div align="center">

<!-- <picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/tess_logo_dark.png">
  <img alt="TESS" src="assets/tess_logo_light.png" height="80">
</picture> -->

<h1>TESS</h1>
<h3>From Text to Forecasts: Bridging Modality Gap with<br/>Temporal Evolution Semantic Space</h3>

<p><img src="https://img.shields.io/badge/🎙️_ICML_2026-Oral_Presentation-FF0000?style=for-the-badge" /></p>

<p>
  <a href="https://arxiv.org"><img src="https://img.shields.io/badge/ICML%202026-Oral%20Paper-FF0000?style=flat-square&logo=googledocs&logoColor=white" /></a>
  &nbsp;
  <a href="https://github.com/olivia3395/TESS"><img src="https://img.shields.io/badge/GitHub-Code-181717?style=flat-square&logo=github" /></a>
  &nbsp;
  <a href="#"><img src="https://img.shields.io/badge/Poster-PDF-FF6B35?style=flat-square&logo=adobeacrobatreader&logoColor=white" /></a>
  &nbsp;
  <img src="https://img.shields.io/badge/Python-3.8+-3776AB?style=flat-square&logo=python&logoColor=white" />
  &nbsp;
  <img src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" />
  &nbsp;
  <img src="https://img.shields.io/badge/License-MIT-22C55E?style=flat-square" />
</p>

<p>
<b>Lehui Li*</b> &nbsp;·&nbsp; <b>Yuyao Wang*</b> &nbsp;·&nbsp; <b>Jisheng Yan*</b> &nbsp;·&nbsp; Wei Zhang &nbsp;·&nbsp; Jinliang Deng† &nbsp;·&nbsp; Haoliang Sun &nbsp;·&nbsp; Zhongyi Han &nbsp;·&nbsp; Yongshun Gong†
</p>


<br/>

> *"Instead of forcing models to read between the lines,<br/>we rewrite the lines into a language models already speak."*

</div>



## What is TESS?

Real-world time series are constantly disrupted by external events — breaking news, weather shocks, policy announcements, market sentiment. These events leave traces in text *long before* they appear in numbers. Yet existing multimodal forecasters fail to harness this: they feed raw text embeddings directly into numerical models, creating a fundamental **modality gap**.

<br/>

<div align="center">
<table>
<tr>
<td align="center" width="33%">

**❌ The Problem**<br/><br/>
Models over-attend to redundant tokens.<br/>
Qualitative language resists numerical decoding.<br/>
Direct fusion causes unstable, noisy training.

</td>
<td align="center" width="5%">→</td>
<td align="center" width="33%">

**💡 Our Insight**<br/><br/>
Don't fuse raw text — *translate* it first.<br/>
Extract structured temporal signals that<br/>forecasters can actually reason over.

</td>
<td align="center" width="5%">→</td>
<td align="center" width="33%">

**✅ TESS**<br/><br/>
An LLM distills text into 4 temporal primitives.<br/>
A gating network filters noisy extractions.<br/>
Primitives condition forecasting as prefix tokens.

</td>
</tr>
</table>
</div>



## Method

<div align="center">
<img src="assets/tess_overview.png" width="88%" alt="TESS Overview"/>
<br/><br/>
<sub>An LLM extracts temporal primitives via structured prompting · Confidence-aware gating filters unreliable signals · Primitives condition a PatchTST forecaster as semantic prefix tokens</sub>
</div>

<br/>

TESS operates in three stages:

**① Build the Semantic Space** &nbsp;—&nbsp; We define four *Temporal Semantic Primitives* (TSPs) that describe how a time series evolves. Each primitive is numerically verifiable from the actual forecast window, enabling reliable training supervision.

<br/>

<div align="center">

| Primitive | What It Captures | Labels |
|:---:|:---|:---|
| 📈 **Distribution Shift** | Mean level change between history and forecast | `STRONG-RISE` `MILD-RISE` `STABLE` `MILD-DROP` `STRONG-DROP` |
| 〰️ **Volatility Shift** | Change in variance / fluctuation intensity | `STRONG-RISE` `MILD-RISE` `STABLE` `MILD-DROP` `STRONG-DROP` |
| 🔀 **Shape** | Overall morphology of the forecast trajectory | `ASCEND` `DESCEND` `PEAK` `TROUGH` `OSCILLATE` |
| ⏱️ **Lag & Decay** | When the event impact begins and how long it lasts | `EARLY-FADE` `EARLY-PERSIST` `MID-FADE` `MID-PERSIST` `LATE` `DIFFUSE` |

</div>

<br/>

**② Text → Primitives** &nbsp;—&nbsp; A frozen LLM classifies each primitive from input text under a structured prompt. A **confidence-aware gating network** estimates extraction reliability from the LLM's own probability margin, softly suppressing unreliable primitives at inference time.

**③ Primitives → Forecast** &nbsp;—&nbsp; Gated primitive embeddings are prepended as **semantic prefix tokens** alongside patch embeddings inside a Transformer. Semantic signals participate in every attention layer. The full model trains end-to-end with a joint forecasting + gating loss (LLM frozen).


## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/olivia3395/TESS.git && cd TESS

pip install torch>=2.0.0 transformers accelerate
pip install numpy pandas scikit-learn einops
pip install -r requirements.txt
```



## Hyperparameters

| Parameter | Description | Default |
|:---|:---|:---:|
| `--lambda_gate` | Gating supervision loss weight | `0.1` |
| `--temperature` | LLM softmax temperature | `1.0` |
| `--patch_len` | Patch length | `16` |
| `--stride` | Patch stride | `8` |
| `--d_model` | Transformer hidden dimension | `128` |
| `--n_heads` | Attention heads | `8` |
| `--n_layers` | Transformer encoder layers | `3` |
| `--lr` | Learning rate (AdamW) | `1e-4` |
| `--patience` | Early stopping patience | `10` |
| `--llm_model` | LLM backbone for extraction | `Qwen3-8B` |




<div align="center">
<sub>
⭐ If you find TESS useful, please consider starring the repo!
<br/><br/>
<i>ICML 2026 Oral · Shandong University · Boston University · Beihang University</i>
</sub>
</div>
