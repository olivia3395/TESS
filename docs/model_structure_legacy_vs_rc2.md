# Legacy Multimodal Baseline vs RC2 Additive Primitive Model

This note compares:

- `legacy/src/model_trainer/models/multimodal_baseline.py`
- `models/legacy_multimodal_primitive_additive.py`

It focuses on implemented model structure, not training behavior.

## 1. Legacy `MultiModal_Baseline`

### Inputs and Shapes

Implemented in code:

- `x_enc`: documented as `[B, L, Channel_Size]`, but `forward()` immediately applies `x_enc.unsqueeze(-1)`. In practice this implies the caller likely passes `[B, L]`, producing `[B, L, 1]`.
- `news_feat`: optional precomputed text/news feature tensor. Shape is not checked, but `dynamic_fc = Linear(text_emb_dim, embedding_size)` implies `[B, text_emb_dim]`.
- Output: `self.dec_out`, shape `[B, pred_len]` after squeezing the last singleton channel.

Unclear from code:

- Exact provenance of `news_feat`: it is only described as a pre-extracted text representation.
- Whether multi-channel time series are actually supported, because `unsqueeze(-1)` conflicts with the docstring if `x_enc` already has a channel dimension.

### Main Submodules

Implemented:

- `DataEmbedding(enc_in, hid_dim, embed, dropout)`: value convolution plus positional embedding, and optional temporal embedding if marks are supplied. This model calls it without time marks.
- `TempEncoder`: legacy temporal encoder returning `(h_env, h_ts)`.
  - Environment branch: average over multiple causal-ish `Conv1d` kernels.
  - Entity/time-series branch: self-attention in time domain plus learned frequency-domain transform, then dropout.
- `mlp_flatten_2`: `Linear(seq_len * hid_dim -> hid_dim)`, `PReLU`, dropout.
- `dynamic_fc`: maps `news_feat` from `text_emb_dim` to `embedding_size`.
- `CrossModalAttention`: projects one time vector and one text vector to Q/K/V, applies scalar dot-product attention, residual layer norm, output projection.
- `decoder_mlp`: `hid_dim -> 256 -> 512 -> pred_len`.

Defined but not used in `forward()`:

- `mlp_flatten`
- `time_to_mm`
- `h_env` from `TempEncoder`
- `mi_regulization`, `beta1`, `beta2`, `normalize_layers`
- imported LLM/tokenizer classes
- imported `CrossModalTransformer`

### Forward Flow

Let `x in R^{B x L}` after caller input and `d = hid_dim`.

Implemented data flow:

```python
x = x_enc.unsqueeze(-1)              # likely [B, L, 1]
mu = mean(x, dim=1, keepdim=True)
sigma = sqrt(var(x - mu, dim=1) + 1e-5)
x_norm = (x - mu) / sigma

e = DataEmbedding(x_norm)            # [B, L, d]
h_env, h_seq = TempEncoder(e)         # both approximately [B, L, d]
h_ts = mlp_flatten_2(flatten(h_seq))  # [B, d]

if news_feat is not None:
    h_news = dynamic_fc(news_feat)    # [B, d]
    h_ts = CrossModalAttention(h_ts, h_news)

y_norm = decoder_mlp(h_ts)            # [B, pred_len]
y_hat = y_norm[..., None] * sigma + mu
y_hat = squeeze_last_dim(y_hat)       # [B, pred_len]
```

Mathematically:

\[
\tilde{x} = (x - \mu) / \sigma
\]

\[
E = \mathrm{DataEmbedding}(\tilde{x})
\]

\[
(H_{env}, H_{ts}) = \mathrm{TempEncoder}(E), \quad
h_{ts} = \mathrm{MLP}_{flat}(\mathrm{vec}(H_{ts}))
\]

If news features are present:

\[
h_{news} = \mathrm{MLP}_{news}(z_{news})
\]

\[
h = \mathrm{CrossModalAttention}(h_{ts}, h_{news})
\]

Otherwise:

\[
h = h_{ts}
\]

Forecast:

\[
\hat{y}_{norm} = \mathrm{MLP}_{dec}(h), \quad
\hat{y} = \sigma \hat{y}_{norm} + \mu
\]

### How Text/News Enters

Implemented:

- Text/news enters only through `news_feat`, a precomputed dense vector.
- It is projected to model dimension by `dynamic_fc`.
- It modifies the single flattened time-series vector through `CrossModalAttention`.

Important implementation detail:

- `CrossModalAttention` receives two 2D tensors `[B, d]`. Its attention score has shape `[B, 1]`, and `softmax(..., dim=-1)` over one element is always 1. Therefore, despite the name "attention", there is no multi-token attention distribution in this path. The effective operation is closer to:

\[
h = W_o \, \mathrm{LN}(W_v h_{news} + h_{ts})
\]

with Q/K scores recorded but not meaningfully weighting multiple items.

### Final Forecast

Implemented:

- A single decoder MLP maps the fused or unfused vector to `pred_len`.
- The forecast is produced in normalized space and then de-normalized with the input mean and standard deviation.
- Loss helper uses MSE between `self.dec_out[:, -pred_len:]` and the last `pred_len` target values.

Not implemented:

- No patching.
- No PatchTST encoder.
- No semantic prefix tokens.
- No learned gate over text/news in this class.
- No direct LLM call inside `forward()`.

## 2. RC2 `LegacyMultimodalPrimitiveAdditive`

### Inputs and Shapes

Implemented in code:

- `x`: required `[B, L, 1]`; checked explicitly.
- `primitive_ids`: required `[B, 4]`; four primitive categorical IDs.
- `primitive_mask`: optional `[B, 4]`; masks individual primitive embeddings by multiplication.
- Output:
  - default: `y_hat`, shape `[B, pred_len, 1]`.
  - with `return_components=True`: dictionary with `y_hat`, `y_num`, `y_primitive_delta`, `y_num_norm`, and `y_primitive_delta_norm`.

Inferred from constructor:

- Four primitive vocabularies default to `(6, 4, 6, 4)`.
- Each primitive ID is embedded into `primitive_emb_dim`, default 32.

Unclear from code:

- The semantic meaning of the four primitive categories is not defined in this file.
- Whether IDs come from an LLM, a label cache, oracle annotations, or another preprocessing path is outside this model file.

### Main Submodules

Implemented:

- `DataEmbedding(c_in=1, d_model=d_model)`: RC2 local legacy-compatible embedding, value convolution plus positional embedding.
- `TempEncoder`: local port of the legacy temporal encoder.
- `mlp_flatten`: `Linear(seq_len * d_model -> d_model)`, `PReLU`, dropout.
- `numerical_decoder`: `d_model -> 256 -> 512 -> pred_len`.
- `primitive_embeddings`: four separate embedding tables.
- `dynamic_fc`: `Linear(4 * primitive_emb_dim -> d_model)`, `PReLU`, dropout.
- `primitive_decoder`: `d_model -> primitive_decoder_hidden -> pred_len`.
- `text_delta_scale`: scalar multiplier on the primitive forecast delta.

### Forward Flow

Implemented data flow:

```python
mu = mean(x, dim=1, keepdim=True)
sigma = sqrt(var(x - mu, dim=1) + 1e-5)
x_norm = (x - mu) / sigma

e = DataEmbedding(x_norm)                 # [B, L, d_model]
_, h_seq = TempEncoder(e)                 # [B, L, d_model]
h_ts = mlp_flatten(flatten(h_seq))        # [B, d_model]
y_num_norm = numerical_decoder(h_ts)      # [B, pred_len]

p = concat([
    Emb_0(primitive_ids[:, 0]),
    Emb_1(primitive_ids[:, 1]),
    Emb_2(primitive_ids[:, 2]),
    Emb_3(primitive_ids[:, 3]),
])                                        # [B, 4 * primitive_emb_dim]
p = p * primitive_mask where provided
h_primitive = dynamic_fc(p)               # [B, d_model]
y_primitive_delta_norm = primitive_decoder(h_primitive)

y_hat_norm = y_num_norm + text_delta_scale * y_primitive_delta_norm
y_hat = y_hat_norm[..., None] * sigma + mu
```

Mathematically:

\[
\tilde{x} = (x - \mu) / \sigma
\]

\[
E = \mathrm{DataEmbedding}(\tilde{x})
\]

\[
(H_{env}, H_{ts}) = \mathrm{TempEncoder}(E), \quad
h_{ts} = \mathrm{MLP}_{flat}(\mathrm{vec}(H_{ts}))
\]

Numeric branch:

\[
\hat{y}_{num}^{norm} = f_{num}(h_{ts})
\]

Primitive branch:

\[
p_i = \mathrm{Emb}_i(c_i), \quad i=1,\dots,4
\]

\[
h_{prim} = \mathrm{MLP}_{prim}([p_1;p_2;p_3;p_4])
\]

\[
\Delta \hat{y}_{prim}^{norm} = f_{prim}(h_{prim})
\]

Late additive fusion:

\[
\hat{y}^{norm} =
\hat{y}_{num}^{norm}
+ \alpha \Delta \hat{y}_{prim}^{norm}
\]

where \(\alpha =\) `text_delta_scale`.

Final de-normalization:

\[
\hat{y} = \sigma \hat{y}^{norm} + \mu
\]

### How Primitive/Text Information Enters

Implemented:

- The model does not consume raw text or news.
- It consumes four discrete primitive IDs and optional masks.
- Primitive IDs are embedded, concatenated, projected to `d_model`, decoded into a forecast-length residual/delta, and added to the numeric forecast in normalized space.

Implied by naming/comments:

- The primitive path is treated as a migrated replacement for the legacy text/news path.
- The docstring calls it a "primitive embedding path" and "late-additive primitive variant".

Not implemented in the base additive class:

- No cross-modal attention between time-series and primitive features.
- No primitive token sequence inserted before numeric tokens.
- No gate.
- No margin input.
- No distillation loss or teacher-student mechanism.
- No LLM execution.

### Same-File Variants

The same RC2 file also implements variants:

- `LegacyMultimodalPrimitiveAdditiveGate`: adds `primitive_margins [B, 4]`, projects primitive embeddings, margins, and a time context from `h_ts` into per-primitive sigmoid gate weights. It averages these weights into `dynamic_scale` and uses:

\[
\hat{y}^{norm} =
\hat{y}_{num}^{norm}
+ \alpha \, s(x,c,m) \, \Delta \hat{y}_{prim}^{norm}
\]

- `LegacyMultimodalPrimitiveAdditiveSoft`: can replace hard primitive ID embeddings with probability-weighted embedding averages from `primitive_probs [B, 4, V]`, excluding the final UNK row by setting its probability to zero.

These are directly implemented in `legacy_multimodal_primitive_additive.py`, but they are not part of the base `LegacyMultimodalPrimitiveAdditive.forward()` signature except through subclassing.

## 3. Final Comparison

### Structural Change

Legacy baseline:

\[
\text{time series} \rightarrow h_{ts}
\quad;\quad
\text{news feature} \rightarrow h_{news}
\quad;\quad
h = \mathrm{CrossModalAttention}(h_{ts}, h_{news})
\quad;\quad
\hat{y} = f(h)
\]

RC2 additive primitive model:

\[
\text{time series} \rightarrow h_{ts} \rightarrow \hat{y}_{num}
\]

\[
\text{primitive IDs} \rightarrow h_{prim} \rightarrow \Delta \hat{y}_{prim}
\]

\[
\hat{y} = \hat{y}_{num} + \alpha \Delta \hat{y}_{prim}
\]

The key change is from early/vector-level fusion before a shared decoder to late forecast-space addition of a primitive-derived residual. The RC2 base additive model removes the legacy `CrossModalAttention` path and gives the primitive branch its own decoder.

### Relation to Paper-Level TESS Claim

The paper-level phrase "LLM primitive -> gated semantic prefix tokens -> PatchTST-style forecaster" does not match the base RC2 additive implementation in this file.

Directly implemented in `LegacyMultimodalPrimitiveAdditive`:

- Primitive IDs are accepted as already available categorical inputs.
- Primitive embeddings are learned lookup tables.
- The primitive path produces an additive forecast delta.
- The numeric encoder is a legacy `DataEmbedding + TempEncoder + flatten MLP`, not PatchTST.

Only implied or external to this file:

- "LLM primitive": primitive IDs may originate from an LLM-driven pipeline elsewhere, but this model only sees IDs/probabilities.
- "Semantic": the embeddings may correspond to semantic primitive labels, but semantics are not represented as text tokens inside the model.

Not implemented in this model:

- Gated semantic prefix tokens in the base additive class.
- Prefix-token concatenation with patched time-series tokens.
- PatchTST-style patching/Transformer forecasting.
- End-to-end LLM primitive extraction.
- Distillation objective.

Partially implemented in same-file variants:

- A gate exists in `LegacyMultimodalPrimitiveAdditiveGate`, but it is a scalar residual scale derived from primitive embeddings, margins, and time context. It is not a gate over semantic prefix tokens inside a PatchTST forecaster.
- Soft primitive probabilities exist in `LegacyMultimodalPrimitiveAdditiveSoft`, but this is probability-weighted embedding lookup, not LLM-token conditioning inside the forecaster.

### Important Mismatch

At code level, RC2 `legacy_multimodal_primitive_additive.py` is best described as:

> legacy temporal encoder + independent primitive embedding residual decoder + normalized-space additive fusion

It is not, from this file alone:

> LLM primitive tokens prepended to a PatchTST-style forecaster with gated semantic prefix conditioning

The implemented model is therefore a pragmatic primitive-residual migration of the legacy multimodal baseline, not a direct implementation of the full paper-level TESS architecture.
