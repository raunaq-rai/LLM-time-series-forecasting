# Deep Learning Coursework - Raunaq Rai (rsr45)

This repository contains the code, experiments, and results for my M2 Deep Learning coursework at the University of Cambridge. The project explores how large language models (LLMs) can be adapted for scientific time-series forecasting under tight computational constraints.

## 🔍 Project Overview

The goal of the coursework was to fine-tune a pretrained LLM—**Qwen2.5-0.5B-Instruct**—on **Lotka–Volterra** predator-prey population dynamics. These are simulated time-series data capturing cyclical biological interactions. To stay within a fixed compute budget of **10¹⁷ FLOPs**, we used:

- **LLMTIME** for numeric tokenisation of time-series
- **Low-Rank Adaptation (LoRA)** for parameter-efficient fine-tuning
- **Custom FLOP estimation scripts** to track compute usage precisely
- **Autoregressive generation** for forecasting future population trajectories

The core idea was to test whether LLMs, typically trained on text, could be adapted for structured numerical tasks under realistic constraints.

## Usage

```bash
conda env create -f environment.yaml
conda activate m2-coursework
pip install e .
```

