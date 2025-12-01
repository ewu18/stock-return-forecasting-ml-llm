# Stock Return Forecasting using ML and Fine-tuned LLMs

Independent research project (2024–2025) comparing traditional ML models and large language models for short-term stock return forecasting.

## Project Overview

- **Task**: Use rolling trading-day windows of technical indicators to predict short-term future returns.
- **ML baselines**: Logistic Regression, Random Forest, Neural Network.
- **LLM approaches**: zero-shot prompting, few-shot prompting, and fine-tuning (OpenAI GPT-3.5).
- **Evaluation**: Accuracy/F1 (direction), RMSE (return), and market-neutral Sharpe ratio.

## Repository Structure

- `notebooks/` – Jupyter notebooks for data prep, ML baselines, LLM experiments.
- `src/` – Reusable Python modules (data processing, model training, evaluation).
- `results/` – Saved metrics and plots (no raw data).
- `data/` – **Local only, not tracked** (Yahoo Finance data, parquet files).

## Environment

Main packages:
`pandas`, `numpy`, `scikit-learn`, `ta`, `yfinance`, `openai`, `torch` (for FinBERT).

API keys are loaded from environment variables and **must not be committed**.
