# time_series

Self-contained Google Colab notebooks for time series forecasting / classification models — from-scratch PyTorch
re-implementations of recent papers, and zero-shot runs of the current time series foundation models.
Each notebook below was executed top-to-bottom with zero errors and is committed with its outputs (runs took 4 s – 2 min each on an
RTX PRO 6000; they are sized for a Colab T4, which will be slower — not separately tested on a T4). The exception is
`time_series_different_architecture`, which needs private Google Drive data.

## Recent models on a common benchmark (ETTh1, 96 → 96)

All new forecasting notebooks share the **same benchmark**, so the numbers below are directly comparable:
ETTh1 (all 7 channels), standard 12/4/4-month train/val/test borders, `StandardScaler` fit on train,
horizon 96, MSE/MAE on the scaled data averaged over every stride-1 test window (2785 windows) — the
protocol used by TSLib and the papers. Seed 2024, single run.

### Supervised, from-scratch PyTorch (lookback L = 96)

| Notebook | Model | Venue | Params | MSE | MAE | Paper MSE / MAE |
|---|---|---|---:|---:|---:|---:|
| [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/manmeet3591/time_series/blob/main/dlinear_decomposition_linear.ipynb) [`dlinear_decomposition_linear`](dlinear_decomposition_linear.ipynb) | [DLinear](https://arxiv.org/abs/2205.13504) | AAAI 2023 | 18,624 | 0.383 | 0.396 | 0.386 / 0.400 |
| [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/manmeet3591/time_series/blob/main/timesnet_2d_variation.ipynb) [`timesnet_2d_variation`](timesnet_2d_variation.ipynb) | [TimesNet](https://arxiv.org/abs/2210.02186) | ICLR 2023 | 605,479 | 0.399 | 0.418 | 0.384 / 0.402 |
| [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/manmeet3591/time_series/blob/main/itransformer_inverted.ipynb) [`itransformer_inverted`](itransformer_inverted.ipynb) | [iTransformer](https://arxiv.org/abs/2310.06625) | ICLR 2024 | 841,568 | 0.389 | 0.404 | 0.386 / 0.405 |
| [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/manmeet3591/time_series/blob/main/timemixer_multiscale_mixing.ipynb) [`timemixer_multiscale_mixing`](timemixer_multiscale_mixing.ipynb) | [TimeMixer](https://arxiv.org/abs/2405.14616) | ICLR 2024 | 75,497 | 0.384 | 0.399 | 0.375 / 0.400 |
| [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/manmeet3591/time_series/blob/main/timexer_exogenous.ipynb) [`timexer_exogenous`](timexer_exogenous.ipynb) | [TimeXer](https://arxiv.org/abs/2402.19072) | NeurIPS 2024 | 1,782,112 | 0.386 | 0.402 | 0.382 / 0.403 |
| [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/manmeet3591/time_series/blob/main/cyclenet_residual_cycle.ipynb) [`cyclenet_residual_cycle`](cyclenet_residual_cycle.ipynb) | [CycleNet](https://arxiv.org/abs/2409.18479) | NeurIPS 2024 | 99,080 | 0.378 | 0.399 | 0.375 / 0.395 |
| [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/manmeet3591/time_series/blob/main/xpatch_dual_stream.ipynb) [`xpatch_dual_stream`](xpatch_dual_stream.ipynb) | [xPatch](https://arxiv.org/abs/2412.17323) | AAAI 2025 | 143,982 | 0.379 | 0.391 | 0.376 / 0.386 |
| [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/manmeet3591/time_series/blob/main/timekan_kan_frequency.ipynb) [`timekan_kan_frequency`](timekan_kan_frequency.ipynb) | [TimeKAN](https://arxiv.org/abs/2502.06910) | ICLR 2025 | 12,779 | 0.382 | 0.401 | 0.367 / 0.395 |

All eight are within ~0.01 MSE of the published number except TimesNet and TimeKAN, whose gaps are seed variance:
TimesNet gets 0.389 with TSLib's default seed 2021, and TimeKAN's official code itself spans 0.369–0.379 across seeds
(details in each notebook). TimeXer also includes the MS setting (OT target, 6 exogenous channels): 0.057 / 0.182 vs paper 0.057 / 0.181.

### Zero-shot foundation models (no training on ETTh1)

| Notebook | Model | From | Params | MSE (L=512) | MAE (L=512) | MSE / MAE (L=96) | Paper MSE / MAE | Notes |
|---|---|---|---:|---:|---:|---:|---:|---|
| [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/manmeet3591/time_series/blob/main/chronos2_zero_shot.ipynb) [`chronos2_zero_shot`](chronos2_zero_shot.ipynb) | [Chronos-2](https://arxiv.org/abs/2510.15821) | Amazon 2025 | 119.5M | 0.395 | 0.380 | 0.468 / 0.414 | – | univariate; jointly-multivariate mode: **0.371 / 0.375** |
| [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/manmeet3591/time_series/blob/main/timesfm25_zero_shot.ipynb) [`timesfm25_zero_shot`](timesfm25_zero_shot.ipynb) | [TimesFM 2.5](https://arxiv.org/abs/2310.10688) | Google (ICML 2024) | 231.3M | 0.384 | 0.382 | 0.429 / 0.402 | – / 0.45 ¹ | + XReg covariates demo |
| [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/manmeet3591/time_series/blob/main/moirai2_zero_shot.ipynb) [`moirai2_zero_shot`](moirai2_zero_shot.ipynb) | [Moirai 2.0](https://arxiv.org/abs/2511.11698) | Salesforce 2025 | 11.4M | 0.384 | 0.380 | 0.435 / 0.403 | – |  |
| [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/manmeet3591/time_series/blob/main/tirex_zero_shot.ipynb) [`tirex_zero_shot`](tirex_zero_shot.ipynb) | [TiRex](https://arxiv.org/abs/2505.23719) | NeurIPS 2025 | 35.3M | 0.386 | 0.379 | 0.439 / 0.401 | – | xLSTM; T4 uses pure-torch backend |
| [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/manmeet3591/time_series/blob/main/toto_zero_shot.ipynb) [`toto_zero_shot`](toto_zero_shot.ipynb) | [Toto](https://arxiv.org/abs/2505.14766) | Datadog 2025 | 151.3M | 0.415 | 0.394 | 0.561 / 0.462 | 0.382 / 0.381 | joint 7-channel; test stride 8 ² |
| [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/manmeet3591/time_series/blob/main/sundial_zero_shot.ipynb) [`sundial_zero_shot`](sundial_zero_shot.ipynb) | [Sundial](https://arxiv.org/abs/2502.00816) | ICML 2025 | 128.3M | 0.355 | 0.383 | 0.406 / 0.396 | 0.348 / 0.385 ³ | L=2880: **0.338 / 0.377** ³; pins transformers 4.40.1 ⁴ |
| [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/manmeet3591/time_series/blob/main/granite_ttm_zero_shot.ipynb) [`granite_ttm_zero_shot`](granite_ttm_zero_shot.ipynb) | [Granite TTM-r2](https://arxiv.org/abs/2401.03955) | NeurIPS 2024 | 0.8M | 0.363 | 0.389 | 0.430 / 0.420 | 0.364 / – | 5% few-shot: 0.363 / 0.389; L=96 col uses 90-ctx model |

¹ TimesFM 1.0 MAE from the paper (last test window only — not directly comparable). No ETTh1 number is published for 2.5.
² Toto is evaluated on every 8th test window with 64 samples to stay within Colab's time budget.
³ Sundial's paper uses a 2880-step lookback; at that setting this notebook reproduces it (0.338 vs 0.348).
⁴ Sundial's remote code breaks on transformers 5.x, so its notebook installs transformers 4.40.1 — run it in a fresh Colab runtime.

Chronos-2, Moirai 2.0, TiRex and TimesFM 2.5 do not publish ETTh1 MSE/MAE (they report GIFT-Eval / fev-bench), so there is no paper column for them.
Several foundation-model packages pin old torch/numpy versions; those notebooks install with `--no-deps` so Colab's torch is left alone — the
"dependency conflicts" message pip prints in that cell is expected.

### Classification

| Notebook | Model | Venue |
|---|---|---|
| [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/manmeet3591/time_series/blob/main/disms_ts_classification.ipynb) [`disms_ts_classification`](disms_ts_classification.ipynb) | [DisMS-TS](https://arxiv.org/abs/2507.04600) — disentangled multi-scale features | ACM MM 2025 |

| Dataset | ACC (ours) | ACC (paper) | macro-F1 (ours / paper) | MCC (ours / paper) |
|---|---:|---:|---:|---:|
| ArticularyWordRecognition | 94.1 | 98.3 | 94.1 / 98.2 | 0.939 / 0.976 |
| FingerMovements | 51.0 | 61.0 | 50.7 / 60.7 | 0.021 / 0.228 |

⚠️ **Not reproduced.** The notebook follows the official code, but lands below the paper on both UEA datasets
(mean of 3 seeds); FingerMovements is near chance, and removing the paper's two regularisers makes no measurable difference.
Possible reasons (validation protocol, preprocessing not published) are discussed in the notebook.

## Earlier notebooks

| Notebook | Notes |
|---|---|
| [`nsdiff_nonstationary_diffusion`](nsdiff_nonstationary_diffusion.ipynb) | NsDiff (Ye, Xu & Gui, ICML 2025, [arXiv 2505.04278](https://arxiv.org/abs/2505.04278)). Citation fixed; forward-diffusion noise scale, missing variance loss (Eq. 13), γ₂ coefficient (Eq. 12) and EMA warm-up fixed — diffusion-mean MSE 5.39 → 0.66 on the synthetic data. |
| [`finetune_timesfm`](finetune_timesfm.ipynb) | LoRA fine-tuning of TimesFM 2.5 via `transformers` (now upgrades to `transformers>=5.3`, where TimesFM 2.5 was added). Zero-shot counterpart: `timesfm25_zero_shot`. |
| [`dtaf_temporal_frequency`](dtaf_temporal_frequency.ipynb) | DTAF dual-branch temporal-frequency model (synthetic data). |
| [`dst_ddpm_spatio_temporal`](dst_ddpm_spatio_temporal.ipynb) | DST-DDPM spatio-temporal diffusion (random graph data). |
| [`dhmoe_diffusion_moe`](dhmoe_diffusion_moe.ipynb) | DHMoE diffusion + hierarchical MoE (random data). Runs, but full reverse-diffusion sampling diverges (RMSE ~1e5) — known issue. |
| [`lstm_time_series`](lstm_time_series.ipynb) | Standalone PyTorch LSTM demo (forward pass now under `no_grad` so it fits in Colab RAM). |
| [`time_series_different_architecture`](time_series_different_architecture.ipynb) | LSTM (Keras), PatchTST, Autoformer, Informer, TimesFM on the HadGEM DSL → AVISO DSL bias-correction task (needs the Google Drive data; CatBoost section is a placeholder). |

## Links / to try

- DST-DDPM: https://drive.google.com/file/d/1h5ZbRhqOKI05qSyVOiN8-FkIY7r93pwt/view
- Try this Chronos RAG: https://proceedings.neurips.cc/paper_files/paper/2025/file/eed25c037bc08afcbefab6f7a6b700e0-Paper-Conference.pdf
- Try this as well Flamingo timesfm: https://huggingface.co/PartAI/FlaMinGo-timesfm
- Monash time series forecasting archive: https://arxiv.org/pdf/2105.06643
