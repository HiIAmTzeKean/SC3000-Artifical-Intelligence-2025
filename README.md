# SC3000-AI-2025

## Overview

This repository contains a small research / lab assignment project that demonstrates fine-tuning a NanoGPT-style Transformer using Direct Preference Optimization (DPO). The goal is to teach a compact GPT model to solve and explain math problems by training from preference pairs (preferred/correct solutions vs. dispreferred/incorrect ones) rather than using a separate reward model + RLHF pipeline.

- Uses a compact, NanoGPT-like architecture implemented in `model.py` (GPT and a GPTRewardModel wrapper).
- Training workflow and explanation are captured in `DPO_finetuning.ipynb` which implements DPO, explains the math behind it, and includes runnable cells for preparing data, hyperparameters, and training loops.
- `configurator.py` supports simple CLI-style configuration overriding for running scripts with either a config file or --key=value style overrides.
- Pretrained / checkpoint files (binary PyTorch weights) stored in `dpo/dpo.pt` for convenience.

## Repository structure

- `DPO_finetuning.ipynb` — primary notebook that documents the experiment, DPO algorithm, hyperparameters, and includes runnable training / evaluation code (Colab-oriented cells included).
- `model.py` — NanoGPT-like model implementation (GPT, attention blocks, MLPs, reward head helper, and utilities like from_pretrained and configure_optimizers).
- `configurator.py` — small helper to override settings through CLI args or by executing a config file.
- `pyproject.toml` — project metadata and Python dependencies.
- `data/` — (training data) contains preference pairs and related artifacts. Note: large data files may not be committed here; dataset loading in the notebook expects JSON/pairs format.
- `dpo/` — saved DPO checkpoint(s).
- `sft/` — saved SFT / base GPT checkpoint(s).
