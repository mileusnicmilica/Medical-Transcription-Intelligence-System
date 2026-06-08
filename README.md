# Medical Transcription Intelligence System

A hybrid NLP pipeline for automated medical transcription analysis. The system combines classical machine learning with transformer-based models to address three tasks: specialty classification, semantic search, and medical entity extraction.

---

## Table of Contents

- [About the Project](#about-the-project)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Pipeline Overview](#pipeline-overview)
- [Methods](#methods)
  - [1. Data Loading and Cleaning](#1-data-loading-and-cleaning)
  - [2. Text Preprocessing](#2-text-preprocessing)
  - [3. Baseline Classification - TF-IDF + LinearSVC](#3-baseline-classification--tf-idf--linearsvc)
  - [4. Few-Shot Classification - SetFit](#4-few-shot-classification--setfit)
  - [5. Semantic Search - FAISS](#5-semantic-search--faiss)
  - [6. Medical Entity Extraction - LLM via Ollama](#6-medical-entity-extraction--llm-via-ollama)
  - [7. LLM-as-Judge Evaluation](#7-llm-as-judge-evaluation)
- [Results](#results)
- [Discussion](#discussion)
- [Installation and Usage](#installation-and-usage)

---

## About the Project

The goal of this project is to build an intelligent system capable of analyzing medical transcriptions across multiple axes:

- **Classification**: automatically determining the medical specialty of a transcription
- **Semantic search**: finding similar transcriptions based on content
- **Information extraction**: identifying key medical entities (diagnoses, medications, symptoms, procedures)

The system uses a hybrid approach - combining classical ML (TF-IDF + LinearSVC), few-shot transformer-based learning (SetFit), dense vector search (FAISS), and LLM-based extraction and evaluation (via Ollama).

---

## Dataset

**Source**: [Medical Transcriptions - Kaggle (tboyle10)](https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions)

The dataset (`mtsamples.csv`) contains approximately 5,000 real-world anonymized medical transcription samples across 40+ medical specialties. Each record includes a free-text transcription and its associated specialty label.

**Preprocessing decisions:**
- Records with missing transcriptions were removed.
- Specialty label strings were stripped of leading/trailing whitespace (e.g., `" Neurology"` → `"Neurology"`).
- Medical specialties with fewer than 20 samples were excluded to ensure statistical significance during training and evaluation.

After cleaning, the dataset retains **4,841 samples across 29 specialties**.

---

## Project Structure

```
Medical-Transcription-Intelligence-System/
│
├── main.py                    # Entry point - runs full pipeline
├── requirements.txt
├── .gitignore
│
├── src/
│   ├── data_loader.py         # Kaggle download + initial cleaning
│   ├── preprocessor.py        # SpaCy-based text preprocessing
│   ├── balancing.py           # SMOTE class balancing (standalone utility)
│   ├── classifiers.py         # LinearSVC with GridSearchCV
│   ├── evaluator.py           # Metric computation (F1, balanced accuracy)
│   ├── visualizer.py          # Class distribution plots, word clouds
│   ├── utils.py               # Model saving/loading utilities
│   ├── searcher.py            # FAISS semantic search + Precision@K evaluation
│   ├── setfit_classifier.py   # SetFit few-shot classifier
│   └── extractor.py           # LLM-based entity extraction + LLM-as-Judge
│
├── data/                      # Generated at runtime 
│   ├── cleaned_data_cache.csv # Cached SpaCy preprocessing output
│   ├── faiss_index_all-MiniLM-L6-v2/
│   ├── faiss_index_NeuML_pubmedbert-base-embeddings/
│   ├── class_distribution_before_balancing.png
│   └── class_distribution_after_balancing.png
│
└── models/                    # Saved trained models
    ├── setfit_v1/             # SetFit model weights + config
    ├── linear_svc_v1_pipeline.pkl
    └── setfit_v1_label_encoder.pkl
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
SpaCy Preprocessing → cached to data/cleaned_data_cache.csv
      │
      ├──► TF-IDF (max 5000 features) + class_weight='balanced'
      │         │
      │         ▼
      │    LinearSVC + 5-fold Cross-Validation
      │         │
      │         ▼
      │    Evaluation: Weighted F1, Balanced Accuracy
      │
      ├──► Sentence Embeddings (paraphrase-MiniLM-L6-v2)
      │         │
      │         ▼
      │    SetFit Few-Shot Training (16 samples/class)
      │         │
      │         ▼
      │    Evaluation: Weighted F1, Balanced Accuracy
      │
      ├──► Sentence Embeddings (all-MiniLM-L6-v2 / PubMedBERT)
      │         │
      │         ▼
      │    FAISS Index → Semantic Search
      │         │
      │         ▼
      │    Evaluation: Precision@K (K=1,3,5)
      │
      └──► LLM via Ollama (llama3.2:3b)
                │
                ▼
           Entity Extraction (diagnoses, medications, symptoms, procedures)
                │
                ▼
           LLM-as-Judge Quality Evaluation
```

---

## Methods

### 1. Data Loading and Cleaning

**File**: `src/data_loader.py`

The dataset is downloaded automatically using `kagglehub` (version-pinned in `requirements.txt` to avoid compatibility issues with the `get_web_endpoint` API change in recent SDK versions). The raw CSV is loaded into a Pandas DataFrame.

Cleaning steps:
- Drop rows where the `transcription` column is `NaN`.
- Strip whitespace from specialty labels.
- Remove specialties with fewer than 20 samples.

### 2. Text Preprocessing

**File**: `src/preprocessor.py`

A `MedicalPreprocessor` class wraps a SpaCy pipeline (`en_core_web_sm`) to perform:
- Lowercasing
- Tokenization
- Lemmatization (words reduced to their dictionary base form)
- Removal of stop words, punctuation, whitespace tokens, and numeric tokens

The `parser` and `ner` SpaCy components are disabled during preprocessing to reduce computation time, since only lemmatization is needed at this stage.

The preprocessed text is stored in a new column `cleaned_transcription` and cached to `data/cleaned_data_cache.csv` to avoid re-running the slow preprocessing step on every execution. The SpaCy import is placed inside the `else` branch of the cache check so that the model can run without SpaCy installed as long as the cache file exists.

### 3. Baseline Classification - TF-IDF + LinearSVC

**Files**: `src/classifiers.py`, `src/evaluator.py`, `main.py`

A TF-IDF vectorizer (max 5,000 features) combined with a **LinearSVC** classifier (`class_weight='balanced'`) is trained and evaluated using 5-fold **StratifiedKFold** cross-validation.

`class_weight='balanced'` causes the model to penalize misclassifications on underrepresented classes more heavily, which partially compensates for the class imbalance without explicit oversampling.

**Note on balancing**: `src/balancing.py` implements SMOTE over a TF-IDF matrix as a standalone utility. SMOTE was not used in the final pipeline due to memory constraints on the local machine - applying SMOTE on a sparse 5,000-feature matrix across 29 classes requires significant RAM. The class imbalance is instead handled via `class_weight='balanced'` on LinearSVC.

Evaluation metrics:
- **Weighted F1 score** - accounts for class imbalance by weighting each class's F1 by its support
- **Balanced Accuracy** - mean recall across all classes, robust to imbalance

### 4. Few-Shot Classification - SetFit

**File**: `src/setfit_classifier.py`

**SetFit** (Sentence Transformer Fine-Tuning) is a few-shot learning framework that avoids the need for large labeled datasets. It works by:
1. Fine-tuning a Sentence Transformer via contrastive learning on a small set of labeled examples (16 samples per class).
2. Training a lightweight classification head on the resulting embeddings.

**Base model**: `sentence-transformers/paraphrase-MiniLM-L6-v2`

**Data preparation** (`prepare_setfit_data`):
- The full dataset is split into 80% train / 20% test using stratified splitting (`random_state=42`).
- From the training portion, 16 examples per class are sampled for few-shot training.
- From the remaining training examples (not selected for few-shot), up to 8 samples per class are taken to form a **validation set**, which is passed to the `Trainer` as `eval_dataset`. This ensures the test set is never seen during training.

### 5. Semantic Search - FAISS

**File**: `src/searcher.py`

A `MedicalSearcher` class builds a **FAISS** index over the cleaned transcriptions using sentence embeddings. Embeddings are L2-normalized and indexed with `IndexFlatIP` (inner product, equivalent to cosine similarity after normalization).

Two embedding models are compared:
- `all-MiniLM-L6-v2` - general-purpose, fast
- `NeuML/pubmedbert-base-embeddings` - domain-specific, trained on biomedical text

**Evaluation** uses **Precision@K**: for 100 sampled queries, it measures the fraction of the top-K retrieved documents (excluding the query document itself) that share the correct specialty with the query. This is computed for K ∈ {1, 3, 5}.

The query document is excluded from results by fetching `k+1` results and skipping the first, since the query itself is always the nearest neighbor in the index.

### 6. Medical Entity Extraction - LLM via Ollama

**File**: `src/extractor.py`

The `extract_medical_entities` function sends a transcription to a locally running LLM (`llama3.2:3b` via the Ollama API) with a structured prompt requesting extraction of:
- `diagnoses`
- `medications`
- `symptoms`
- `procedures`

Only the first 1,000 characters of each transcription are sent to stay within practical context and latency limits of the small local model. The output is parsed as JSON with a fallback to `raw_response` if parsing fails.

If Ollama is not running locally, `main.py` catches the connection error and falls back to displaying pre-computed sample results from a Colab run.

### 7. LLM-as-Judge Evaluation

**File**: `src/extractor.py`

The quality of each extraction is evaluated by a second LLM call using a system prompt (`JUDGE_SYSTEM_PROMPT`) that instructs the model to act as an expert medical extraction evaluator.

**Scoring formula** (applied per field):

```
correctness  = C / (C + H)   if (C + H) > 0, else 1.0
completeness = C / (C + M)   if (C + M) > 0, else 1.0
field_score  = (correctness + completeness) / 2
overall_score = average of all four field_scores
```

Where C = correctly extracted items, M = missing items, H = hallucinated items.

The judge also returns lists of missing and hallucinated items per field, and one sentence of explanation per field describing the main source of error.

---

## Results

### Baseline: TF-IDF + LinearSVC

| Metric | Cross-Validation (5-fold) | Test Set |
|---|---|---|
| Weighted F1 | 0.1847 ± 0.0034 | 0.1449 |
| Balanced Accuracy | 0.2979 ± 0.0283 | 0.2514 |

### SetFit (Few-Shot, 16 samples/class)

| Metric | Test Set |
|---|---|
| Weighted F1 | 0.2715 |
| Balanced Accuracy | 0.4519 |

### Semantic Search - Precision@K

| Model | Precision@1 | Precision@3 | Precision@5 |
|---|---|---|---|
| all-MiniLM-L6-v2 | 0.45 | 0.4033 | 0.39 |
| NeuML/pubmedbert-base-embeddings | 0.43 | 0.3967 | 0.378 |

### LLM Extraction - Average Judge Score

| Model | Avg. Judge Score (4 samples) |
|---|---|
| llama3.2:3b | 0.873 |

---

## Discussion

**TF-IDF + LinearSVC** achieves relatively low scores (F1: 0.14, balanced accuracy: 0.25) due to a fundamental limitation of bag-of-words representations on this dataset. Medical specialties share significant vocabulary overlap - Surgery transcriptions in particular contain terminology from nearly every other specialty, making it the dominant and most confused class (recall: 0.02 on test set). TF-IDF cannot disambiguate based on context, only on word frequency. This directly motivates the use of transformer-based approaches.

**SetFit** achieves notably better balanced accuracy (0.45 vs 0.25) using only 16 labeled examples per class. The improvement comes from sentence embeddings encoding semantic context rather than raw word frequencies, which helps separate specialties with overlapping vocabulary. The gap in weighted F1 (0.27 vs 0.14) is smaller because weighted F1 is dominated by large classes like Surgery where SetFit also struggles.

**FAISS semantic search** achieves Precision@1 of 0.45 (MiniLM) and 0.43 (PubMedBERT), meaning the nearest neighbor in embedding space is from the correct specialty in roughly half of queries. The general-purpose MiniLM model slightly outperforms the domain-specific PubMedBERT on this dataset, likely because the transcriptions use cleaned/lemmatized text that differs from PubMedBERT's pretraining distribution.

**LLM extraction** achieves an average judge score of 0.873 across 4 sample transcriptions. The model reliably extracts diagnoses and procedures but occasionally misses medications (which are often implicit in the transcription or listed by brand name). The judge scoring formula is explicitly defined in the system prompt to reduce arbitrariness across runs.

**Limitations**: The dataset consists of relatively clean, structured clinical notes rather than raw speech-to-text output. Real-world performance on noisy ASR transcriptions would likely be lower. Additionally, the LLM evaluation pipeline is limited to short transcription windows (1,000 characters) to keep inference practical on CPU.

---

## Installation and Usage

### Requirements

```bash
pip install -r requirements.txt
```

SpaCy model is only needed if running preprocessing from scratch (without cache):
```bash
python -m spacy download en_core_web_sm
```

For LLM-based extraction, install and run [Ollama](https://ollama.ai) locally:
```bash
ollama pull llama3.2:3b
ollama serve
```

### Run the full pipeline

```bash
python main.py
```

This will:
1. Download the dataset via kagglehub
2. Load preprocessed data from cache (or run SpaCy preprocessing if cache missing)
3. Train and evaluate TF-IDF + LinearSVC with cross-validation
4. Load and evaluate the pre-trained SetFit model
5. Build or load FAISS indexes and evaluate semantic search
6. Run entity extraction (or display Colab sample results if Ollama is not running)

### Train SetFit from scratch

```python
from src.setfit_classifier import prepare_setfit_data, train_setfit, save_setfit_model
import pandas as pd

df = pd.read_csv("data/cleaned_data_cache.csv")
train_ds, val_ds, test_ds, le = prepare_setfit_data(df, n_samples=16)
model = train_setfit(train_ds, val_ds, le)
save_setfit_model(model, le)
```

### Run semantic search interactively

```python
from src.searcher import MedicalSearcher
import pandas as pd

df = pd.read_csv("data/cleaned_data_cache.csv")
searcher = MedicalSearcher()
searcher.load()
results = searcher.search("patient presents with chest pain and shortness of breath", k=5)
for r in results:
    print(r['specialty'], r['score'])
```
