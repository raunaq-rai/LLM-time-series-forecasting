# The application of LLMs to time series forecasting
### Raunaq Rai rsr45@cam.ac.uk

The project explores how **Large Language Models (LLMs)** can be adapted to perform **scientific time-series forecasting** under strict computational constraints.

---

## Project Overview

The goal of this coursework was to fine-tune a pretrained LLM—**Qwen2.5-0.5B-Instruct**—on simulated population dynamics from the **Lotka–Volterra equations**, a classic model of predator-prey interactions.

To remain within a compute budget of **10¹⁷ FLOPs**, the following techniques were used:

- **LLMTIME**: Tokenisation of numeric time-series into LLM-friendly formats
- **Low-Rank Adaptation (LoRA)**: Efficient fine-tuning by training only low-rank adapters
- **Custom FLOP Estimator**: Tracks exact compute usage for each experiment
- **Autoregressive Forecasting**: Predicts future population states token by token

This project demonstrates that LLMs, when adapted thoughtfully, can be competitive tools for numerical prediction tasks—despite being originally trained on text.

---

## Installation & Usage

Clone the repository and create the conda environment:

```bash
conda env create -f environment.yaml
conda activate m2-coursework
pip install -e .
pip install -r requirements.txt
pip install ipykernel
python -m ipykernel install --user --name=m2_coursework --display-name "M2 Coursework"
```

---

## Contents
- **src**: Core implementation (LoRA, preprocessing, training).
- **notebooks**: Jupyter notebooks for experiments and evaluation.
- **report**: Report and figures.
- **lotka_volterra_data.h5**: Dataset of predator-prey sequences
- **tests**: test suite for core functionality - use ```pytest tests/```
- **environment.yaml**: Environment setup
- **LICENCE**: MIT Licence
- **pyproject.toml**: Modern package configuration
- **requirements.txt**: pip usage
- **Flops_tracker.xlsx**: excel file tracking flop usage throughout project

---

## Results Summary

- Final model trained with LoRA rank 8, learning rate = 1e-4, context length = 768
- Validation loss: 0.2981
- Sample forecasting $R^2$ scores: 
    - Prey: 0.88
    - Predator: 0.86
- Total compute used: 88.3% of FLOP budget

