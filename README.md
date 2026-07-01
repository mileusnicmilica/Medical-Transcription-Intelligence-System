# Medical Transcription Intelligence System

A hybrid NLP pipeline for automated medical transcription analysis. The system combines classical machine learning with transformer-based models to address three core tasks: specialty classification, semantic search, and medical entity extraction, with LLM-as-a-judge evaluation.

---

## Table of Contents

- [About the Project](#about-the-project)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Pipeline Overview](#pipeline-overview)
- [Methods](#methods)
  - [1. Data Loading and Cleaning](#1-data-loading-and-cleaning)
  - [2. Text Preprocessing](#2-text-preprocessing)
  - [3. Classification - TF-IDF + LinearSVC](#3-classification--tf-idf--linearsvc)
  - [4. Few-Shot Classification - SetFit](#4-few-shot-classification--setfit)
  - [5. Semantic Search - FAISS](#5-semantic-search--faiss)
  - [6. Medical Entity Extraction - LLM via Ollama](#6-medical-entity-extraction--llm-via-ollama)
  - [7. LLM-as-Judge Evaluation](#7-llm-as-judge-evaluation)
- [Results](#results)
- [Installation and Usage](#installation-and-usage)

---

## About the Project

The goal of this project is to build an intelligent system capable of analyzing medical transcriptions across three axes:

- **Classification**: automatically determining the medical specialty of a transcription using two approaches — TF-IDF + LinearSVC (classical ML) and SetFit (few-shot transformer-based learning)
- **Semantic search**: finding similar transcriptions based on semantic content using dense vector search (FAISS)
- **Information extraction**: identifying key medical entities (diagnoses, medications, symptoms, procedures) using a locally hosted LLM via Ollama, with quality evaluation via a heterogeneous LLM-as-a-judge panel

The system includes a Streamlit demo app (`main_app.py`) that exposes all three components through an interactive UI.

---

## Dataset

**Source**: [Medical Transcriptions - Kaggle (tboyle10)](https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions)

The dataset (`mtsamples.csv`) contains approximately 5,000 real-world anonymized medical transcription samples across 40+ medical specialties.

**Preprocessing decisions:**
- Records with missing transcriptions were removed (33 samples).
- Specialty label strings were stripped of leading/trailing whitespace (e.g., `" Neurology"` → `"Neurology"`).
- Medical specialties with fewer than 20 samples were excluded to ensure statistical significance during training and evaluation.

After cleaning, the dataset retains **4,841 samples across 29 specialties**.

An additional experiment was conducted with the **Surgery class removed** (3,753 samples, 28 specialties), motivated by both quantitative arguments (Surgery dominates the class distribution at 22.5% of samples, distorting macro-averaged metrics) and conceptual arguments (Surgery is not a narrow specialty but a cross-domain surgical approach, leading to high intra-class terminological heterogeneity). The models trained without Surgery are used in the Streamlit demo.

---

## Project Structure

```
Medical-Transcription-Intelligence-System/
│
├── main.py                    # Entry point — runs full pipeline
├── main_app.py                # Streamlit demo application
├── requirements.txt
├── .gitignore
│
├── src/
│   ├── data_loader.py         # Kaggle download + initial cleaning
│   ├── preprocessor.py        # SpaCy-based text preprocessing (MedicalPreprocessor)
│   ├── balancing.py           # SMOTE class balancing (standalone utility, not used in final pipeline)
│   ├── classifiers.py         # LinearSVC with GridSearchCV
│   ├── evaluator.py           # Metric computation (weighted F1, macro F1, balanced accuracy)
│   ├── visualizer.py          # Class distribution plots, word clouds
│   ├── utils.py               # Model saving/loading utilities (joblib)
│   ├── searcher.py            # FAISS semantic search + Precision@K evaluation
│   ├── setfit_classifier.py   # SetFit few-shot classifier
│   └── extractor.py           # LLM-based entity extraction + LLM-as-Judge evaluation
│
├── data/                      # Generated at runtime
│   ├── cleaned_data_cache.csv
│   ├── faiss_index_all-MiniLM-L6-v2_transcription/
│   ├── faiss_index_all-MiniLM-L6-v2_cleaned_transcription/
│   ├── faiss_index_NeuML_pubmedbert-base-embeddings_transcription/
│   ├── faiss_index_NeuML_pubmedbert-base-embeddings_cleaned_transcription/
│   └── class_distribution_*.png
│
└── models/                    # Saved trained models
    ├── setfit_v1/
    ├── setfit_v1_lemmatized/
    ├── setfit_v1_lemmatized_no_surgery/
    ├── setfit_v1_raw/
    ├── linear_svc_v1_pipeline.pkl
    ├── linear_svc_v1_no_surgery_pipeline.pkl
    ├── setfit_v1_label_encoder.pkl
    ├── setfit_v1_lemmatized_label_encoder.pkl
    ├── setfit_v1_lemmatized_no_surgery_label_encoder.pkl
    └── setfit_v1_raw_label_encoder.pkl
```

---

## Pipeline Overview

```
Raw Data (Kaggle)
      │
      ▼
Data Cleaning (data_loader.py)
      │
      ▼
SpaCy Preprocessing (MedicalPreprocessor) → cached to data/cleaned_data_cache.csv
      │
      ├──► TF-IDF (max 5,000 features) + class_weight='balanced'
      │         │
      │         ▼
      │    LinearSVC + GridSearchCV (C ∈ {0.1, 1, 10}, scoring=f1_macro)
      │         │
      │         ├── Full dataset (29 specialties)
      │         └── Without Surgery (28 specialties) ← used in Streamlit demo
      │
      ├──► SetFit (paraphrase-MiniLM-L6-v2, 16 samples/class, 1 iteration)
      │         │
      │         ├── Lemmatized text (cleaned_transcription)
      │         └── Raw text (transcription)  ← ablation study
      │
      ├──► Sentence Embeddings → FAISS IndexFlatIP (cosine similarity)
      │         │
      │         ├── all-MiniLM-L6-v2 [raw] / [lemmatized]
      │         └── NeuML/pubmedbert-base-embeddings [raw] / [lemmatized]
      │                   │
      │                   ▼
      │              Evaluation: Precision@K (K=1,3,5), 100 queries
      │
      └──► LLM via Ollama (llama3.2:3b)
                │
                ▼
           Entity Extraction (diagnoses, medications, symptoms, procedures)
                │
                ▼
           LLM-as-Judge Evaluation
                ├── Self-eval baseline (Ollama judges itself)
                ├── Single Groq judge (openai/gpt-oss-120b)
                └── Heterogeneous panel (gpt-oss-120b + qwen/qwen3.6-27b)
```

---

## Methods

### 1. Data Loading and Cleaning

**File**: `src/data_loader.py`

Dataset is downloaded automatically via `kagglehub`. Cleaning: remove NaN transcriptions, strip specialty label whitespace, remove specialties with fewer than 20 samples.

### 2. Text Preprocessing

**File**: `src/preprocessor.py`

`MedicalPreprocessor` wraps SpaCy (`en_core_web_sm`) to perform lowercasing, tokenization, lemmatization, and removal of stop words, punctuation, whitespace tokens, and numeric tokens. The `parser` and `ner` components are disabled for speed. Results are cached to `data/cleaned_data_cache.csv`.

### 3. Classification — TF-IDF + LinearSVC

**Files**: `src/classifiers.py`, `main.py`

`sklearn.Pipeline` combining TF-IDF (`max_features=5000`) and `LinearSVC` (`class_weight='balanced'`, `random_state=42`, `max_iter=5000`). Hyperparameter tuning via `GridSearchCV` (`C ∈ {0.1, 1, 10}`, `cv=3`, `scoring=f1_macro`). Optimal C=0.1 selected for both full-dataset and no-Surgery variants. Stratified 80/20 train/test split, 5-fold cross-validation as diagnostic only.

`RandomOverSampler` also tested as alternative balancing strategy; `class_weight='balanced'` outperformed it across all metrics. SMOTE implemented in `src/balancing.py` but excluded from final pipeline due to prohibitive runtime on sparse high-dimensional TF-IDF features.

### 4. Few-Shot Classification — SetFit

**File**: `src/setfit_classifier.py`

SetFit fine-tunes `paraphrase-MiniLM-L6-v2` via contrastive learning on 16 samples per class, then trains a classification head on the resulting embeddings. Evaluated in two ablation variants: lemmatized vs. raw text. Both variants trained from scratch with identical configuration (`num_iterations=1`, `num_epochs=1`, `batch_size=16`).

### 5. Semantic Search — FAISS

**File**: `src/searcher.py`

`MedicalSearcher` generates L2-normalized sentence embeddings and indexes them with `IndexFlatIP` (exact cosine similarity search). Four configurations evaluated: 2 models × 2 text variants. Evaluated with Precision@K (K∈{1,3,5}) on 100 sampled queries.

### 6. Medical Entity Extraction — LLM via Ollama

**File**: `src/extractor.py`

`llama3.2:3b` (local, via Ollama REST API) extracts structured JSON with four entity categories. Input truncated to 1,000 characters. JSON parsing with fallback to `raw_response` on failure.

### 7. LLM-as-Judge Evaluation

**File**: `src/extractor.py`

Three evaluation methodologies compared on the same 5 samples (`random_state=42`):

| Methodology | Avg. Score |
|---|---|
| Self-eval baseline (Ollama judges itself) | 0.780 |
| Single Groq judge (openai/gpt-oss-120b) | 0.901 |
| Heterogeneous panel (gpt-oss-120b + qwen/qwen3.6-27b) | 0.797 |

Scoring formula per field: `field_score = (correctness + completeness) / 2`, where `correctness = C/(C+H)` and `completeness = C/(C+M)`. `overall_score` = average of four field scores.

Note: `qwen/qwen3.6-27b` requires `reasoning_effort="none"` to avoid token exhaustion before generating valid JSON output.

---

## Results

### Classification

| Variant | Weighted F1 | Macro F1 | Balanced Acc. |
|---|---|---|---|
| TF-IDF + LinearSVC (full dataset) | 0.254 | 0.370 | 0.532 |
| TF-IDF + LinearSVC (no Surgery) | 0.402 | 0.453 | 0.554 |
| SetFit lemmatized (full dataset) | 0.268 | 0.309 | 0.435 |
| SetFit lemmatized (no Surgery) | 0.384 | 0.395 | 0.489 |
| TF-IDF + RandomOverSampler | 0.227 | 0.326 | 0.433 |
| SetFit raw text (full dataset) | 0.257 | 0.297 | 0.423 |

**Key finding**: Surgery class removal yields larger macro F1 improvement (+0.08 TF-IDF, +0.09 SetFit) than model choice. TF-IDF + LinearSVC consistently outperforms SetFit, likely due to SetFit's constrained training budget (16 samples/class, 1 iteration).

### Semantic Search

| Configuration | Precision@1 | Precision@3 | Precision@5 |
|---|---|---|---|
| MiniLM + lemmatized | **0.450** | **0.403** | **0.390** |
| MiniLM + raw | 0.430 | 0.383 | 0.372 |
| PubMedBERT + lemmatized | 0.430 | 0.397 | 0.378 |
| PubMedBERT + raw | 0.430 | 0.373 | 0.360 |

**Key finding**: Lemmatization consistently improves performance. General-purpose MiniLM outperforms domain-specific PubMedBERT, likely because MTSamples contains substantial everyday clinical language beyond specialized biomedical terminology.

---

## Installation and Usage

### Requirements

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

For entity extraction, install and run [Ollama](https://ollama.ai) locally:
```bash
ollama pull llama3.2:3b
ollama serve
```

For LLM-as-judge evaluation, set your Groq API key:
```bash
# Windows PowerShell
$env:GROQ_API_KEY="your_api_key"

# Linux/macOS
export GROQ_API_KEY="your_api_key"
```

### Run the full pipeline

```bash
python main.py
```

### Run the Streamlit demo

```bash
streamlit run main_app.py
```

The demo runs at `http://localhost:8501`. Classification and semantic search tabs work without Ollama/Groq. The entity extraction tab requires Ollama running locally; the judge evaluation additionally requires `GROQ_API_KEY`.

---

## GitHub Repository

[https://github.com/mileusnicmilica/Medical-Transcription-Intelligence-System](https://github.com/mileusnicmilica/Medical-Transcription-Intelligence-System)
