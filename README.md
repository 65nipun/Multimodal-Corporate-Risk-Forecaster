# Multimodal Corporate Risk Forecaster 📈🎙️

A PyTorch-based machine learning pipeline that forecasts forward-looking equity volatility by fusing unstructured financial text and acoustic prosody (CEO vocal stress) from earnings calls.

## Overview
Traditional financial risk models rely strictly on quantitative metrics or NLP sentiment analysis. This project introduces a **multimodal approach**, hypothesizing that *how* executives speak (vocal micro-tremors, pacing) carries as much predictive signal as *what* they say. 

The pipeline extracts raw audio and transcripts from the Hugging Face `Revai/earnings21` dataset, processes them through specialized encoders, and evaluates them using strict Walk-Forward Cross-Validation to eliminate lookahead bias in the time-series predictions.

## Architecture & Tech Stack
*   **NLP Pipeline (Text):** Hugging Face `ProsusAI/finbert` extracts a 3-dimensional financial sentiment vector (Positive, Negative, Neutral).
*   **Acoustic Pipeline (Audio):** `librosa` extracts Mel-Frequency Cepstral Coefficients (MFCCs) from raw waveforms. The temporal prosody is modeled using a 2-layer **Bidirectional LSTM**.
*   **Target Generation:** The `yfinance` API dynamically builds the ground-truth labels (7-day forward log-return volatility).
*   **Fusion Layer:** Explored both simple vector concatenation and a PyTorch **Cross-Attention** mechanism (`embed_dim=64`, 4 heads) to capture verbal-vocal coherence.

## Ablation Study & Empirical Results
To validate the architecture, the cross-attention model was evaluated against a baseline concatenation approach using Walk-Forward Cross-Validation.

| Model Architecture | Mean RMSE (Lower is Better) |
| :--- | :--- |
| **Baseline (Feature Concatenation)** | **0.004911** |
| Cross-Attention (4-Head) | 0.149533 |

### Limitations & Data Constraints
The empirical results demonstrate that the simpler concatenation baseline significantly outperformed the cross-attention mechanism. Analysis of the walk-forward folds reveals the attention layer was heavily data-starved on the limited sample size ($N=44$). The cross-attention error decayed rapidly as the expanding window grew (0.230 $\to$ 0.191 $\to$ 0.026), indicating that while the multi-head attention architecture was learning, the parameterized complexity required a much larger dataset to converge fully. The simpler baseline generalized reliably within the low-data regime.

## Usage
1. Clone the repository and install dependencies: `pip install -r requirements.txt`
2. Run the pipeline via the provided Jupyter Notebook or by executing the modular scripts in `src/`.
