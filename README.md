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

*Shandong University · Boston University · Beihang University*

<br/>

[![Paper](https://img.shields.io/badge/📄_Paper-ICML_2026-blue?style=for-the-badge)](https://arxiv.org)
[![Code](https://img.shields.io/badge/💻_Code-GitHub-black?style=for-the-badge)](https://github.com/olivia3395/TESS)
[![Poster](https://img.shields.io/badge/🖼️_Poster-PDF-orange?style=for-the-badge)](#)

</div>

<br/>



## 🔍 Overview

Real-world time series are often disrupted by external events — news, weather shocks, market sentiment — that cause abrupt, unpredictable shifts. While text naturally describes these events, existing multimodal methods fail to translate qualitative language into reliable numerical forecasting signals.

**TESS** introduces a **Temporal Evolution Semantic Space** as an intermediate bottleneck: instead of feeding raw text directly to a forecaster, an LLM distills it into compact, interpretable temporal primitives, which then condition the forecasting model as structured exogenous signals.

<br/>

<table>
  <thead>
    <tr>
      <th>⚠️ Challenge</th>
      <th>✅ TESS's Solution</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Models over-attend to redundant text tokens</td>
      <td>Distill text into compact temporal primitives via LLM</td>
    </tr>
    <tr>
      <td>Qualitative language resists numerical decoding</td>
      <td>Define numerically grounded primitives (shift, volatility, shape, lag)</td>
    </tr>
    <tr>
      <td>LLM extraction can be noisy or unreliable</td>
      <td>Confidence-aware gating suppresses unreliable primitives</td>
    </tr>
    <tr>
      <td>Direct text fusion causes unstable training</td>
      <td>Semantic prefix tokens enable smooth, fast convergence</td>
    </tr>
  </tbody>
</table>

<br/>



## Method: Three-Stage Pipeline

<div align="center">
<img src="assets/tess_overview.png" width="90%" alt="TESS Overview"/>
<br/><br/>
<sub><b>Figure 1:</b> Overview of TESS. Text is distilled into temporal primitives by a frozen LLM, filtered by confidence-aware gating, and injected as semantic prefix tokens into a Transformer forecaster.</sub>
</div>

<br/>

### Stage 1 · Temporal Evolution Semantic Space

We define four **Temporal Semantic Primitives (TSPs)** — each verifiable from the actual forecast window, providing reliable training supervision:

<br/>

<table>
  <tbody>
    <tr>
      <td width="30"><b>📐</b></td>
      <td width="160"><b>Distribution Shift</b></td>
      <td><i>Mean level change between history and forecast window</i><br/><br/>
        <code>STRONG-RISE</code> &nbsp;·&nbsp;
        <code>MILD-RISE</code> &nbsp;·&nbsp;
        <code>STABLE</code> &nbsp;·&nbsp;
        <code>MILD-DROP</code> &nbsp;·&nbsp;
        <code>STRONG-DROP</code>
      </td>
    </tr>
    <tr><td colspan="3"><hr/></td></tr>
    <tr>
      <td><b>〰️</b></td>
      <td><b>Volatility Shift</b></td>
      <td><i>Change in variance and fluctuation intensity</i><br/><br/>
        <code>STRONG-RISE</code> &nbsp;·&nbsp;
        <code>MILD-RISE</code> &nbsp;·&nbsp;
        <code>STABLE</code> &nbsp;·&nbsp;
        <code>MILD-DROP</code> &nbsp;·&nbsp;
        <code>STRONG-DROP</code>
      </td>
    </tr>
    <tr><td colspan="3"><hr/></td></tr>
    <tr>
      <td><b>📈</b></td>
      <td><b>Shape</b></td>
      <td><i>Overall morphology of the forecast trajectory</i><br/><br/>
        <code>ASCEND</code> &nbsp;·&nbsp;
        <code>DESCEND</code> &nbsp;·&nbsp;
        <code>PEAK</code> &nbsp;·&nbsp;
        <code>TROUGH</code> &nbsp;·&nbsp;
        <code>OSCILLATE</code>
      </td>
    </tr>
    <tr><td colspan="3"><hr/></td></tr>
    <tr>
      <td><b>⏳</b></td>
      <td><b>Lag &amp; Decay</b></td>
      <td><i>When the event impact begins and how long it persists</i><br/><br/>
        <code>EARLY-FADE</code> &nbsp;·&nbsp;
        <code>EARLY-PERSIST</code> &nbsp;·&nbsp;
        <code>MID-FADE</code> &nbsp;·&nbsp;
        <code>MID-PERSIST</code> &nbsp;·&nbsp;
        <code>LATE</code> &nbsp;·&nbsp;
        <code>DIFFUSE</code>
      </td>
    </tr>
  </tbody>
</table>

<br/>

### Stage 2 · Text → Temporal Semantic Primitives

A frozen LLM classifies each primitive from input text under a structured prompt. A **confidence-aware gating network** then estimates extraction reliability — suppressing noisy primitives during inference via soft weighting.

### Stage 3 · Semantic Primitives-Conditioned Forecasting

Gated primitive embeddings are prepended as **semantic prefix tokens** to the patch embeddings of a PatchTST backbone. This allows semantic signals to participate in all Transformer attention layers. The model trains end-to-end (LLM frozen) with a joint forecasting + gating supervision loss.



## Quick Start

### Installation


### Running Experiments



### Primitive Extraction (Offline Preprocessing)



## ⚙️ Key Hyperparameters

<br/>

<table>
  <thead>
    <tr>
      <th>Parameter</th>
      <th>Description</th>
      <th align="center">Default</th>
    </tr>
  </thead>
  <tbody>
    <tr><td><code>--lambda_gate</code></td><td>Weight for gating supervision loss</td><td align="center"><code>0.1</code></td></tr>
    <tr><td><code>--temperature</code></td><td>LLM softmax temperature for primitive extraction</td><td align="center"><code>1.0</code></td></tr>
    <tr><td><code>--patch_len</code></td><td>Patch length</td><td align="center"><code>16</code></td></tr>
    <tr><td><code>--stride</code></td><td>Patch stride</td><td align="center"><code>8</code></td></tr>
    <tr><td><code>--d_model</code></td><td>Transformer hidden dimension</td><td align="center"><code>128</code></td></tr>
    <tr><td><code>--n_heads</code></td><td>Number of attention heads</td><td align="center"><code>8</code></td></tr>
    <tr><td><code>--n_layers</code></td><td>Number of Transformer encoder layers</td><td align="center"><code>3</code></td></tr>
    <tr><td><code>--epoch</code></td><td>Training epochs</td><td align="center"><code>100</code></td></tr>
    <tr><td><code>--patience</code></td><td>Early stopping patience</td><td align="center"><code>10</code></td></tr>
    <tr><td><code>--lr</code></td><td>Learning rate (AdamW)</td><td align="center"><code>1e-4</code></td></tr>
    <tr><td><code>--llm_model</code></td><td>LLM for primitive extraction</td><td align="center"><code>Qwen3-8B</code></td></tr>
  </tbody>
</table>

<br/>

> 💡 **Tip:** Use `--use_cv True` for automatic hyperparameter selection via cross-validation.
