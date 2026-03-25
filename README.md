# NLU Assignment 2 (B23CM1016)

This repository contains complete code and outputs for:

- **Problem 1**: Word2Vec learning on IIT Jodhpur corpus (from scratch + reference comparison)
- **Problem 2**: Character-level name generation (RNN, BiLSTM, Attention+RNN)

## Project Structure

- `dataset/raw/` : raw IITJ text files
- `src/problem1_pipeline.py` : corpus curation + Word2Vec experiments + analysis + visualization
- `src/problem2_pipeline.py` : dataset generation + RNN variants + evaluation
- `corpus.txt` : cleaned corpus (generated)
- `TrainingNames.txt` : 1000-name dataset for Problem 2 (generated)
- `outputs/` : JSON results, generated samples, and plots

## Environment

Python virtual environment configured at `.venv`.

Install dependencies (already installed in this workspace run):

- numpy
- matplotlib
- scikit-learn
- wordcloud
- tqdm
- pandas
- torch

## Run

### Problem 1

```bash
/Users/faizimdad/Desktop/NLU_PA2/.venv/bin/python src/problem1_pipeline.py
```

### Problem 2

```bash
/Users/faizimdad/Desktop/NLU_PA2/.venv/bin/python src/problem2_pipeline.py
```

## Main Outputs

- `outputs/problem1_results.json`
- `outputs/problem2_results.json`
- `outputs/wordcloud.png`
- `outputs/pca_sgns.png`
- `outputs/pca_cbow.png`
- `outputs/generated_rnn.txt`
- `outputs/generated_bilstm.txt`
- `outputs/generated_attn_rnn.txt`
