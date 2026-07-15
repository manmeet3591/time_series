# time_series

  - lstm_time_series.ipynb — Standalone PyTorch LSTM (random data demo, input→target mapping)
  
  - time_series_different_architecture.ipynb — The main notebook with 5 models all applied to the same climate dataset
  (HadGEM DSL → AVISO DSL bias correction):

    a. LSTM (Keras)
    
    b. PatchTST (PyTorch, from scratch)

    c. Autoformer (PyTorch, from scratch)

    d. Informer (PyTorch, from scratch)

    e. TimesFM (Google's foundation model, fine-tuned)

    f. CatBoost (placeholder, empty cells)

    DST-DDPM: https://drive.google.com/file/d/1h5ZbRhqOKI05qSyVOiN8-FkIY7r93pwt/view

    Try this Chronos RAG: https://proceedings.neurips.cc/paper_files/paper/2025/file/eed25c037bc08afcbefab6f7a6b700e0-Paper-Conference.pdf

    Try this as well Flamingo timesfm: https://huggingface.co/PartAI/FlaMinGo-timesfm

    Monash time series forecasting archive: https://arxiv.org/pdf/2105.06643

