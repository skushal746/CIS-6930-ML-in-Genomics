# CIS-6930 — Machine Learning in Genomics

A multi-modal deep learning classifier for automating the interpretation of genetic mutations in cancer treatment. Built for the **CIS-6930: Machine Learning in Genomics** course at the University of Florida.

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [System Design](#system-design)
- [Feature Engineering](#feature-engineering)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Architecture Decisions](#architecture-decisions)
- [API Contracts](#api-contracts)
- [Setup & Usage](#setup--usage)
- [Known Limitations](#known-limitations)

---

## Problem Statement

Clinical oncologists must manually interpret thousands of genetic mutations to determine which ones drive tumor growth and which are passenger (benign) events. This interpretation requires reading large volumes of clinical literature for each mutation — a process that is slow, expensive, and difficult to scale.

This project automates that classification using machine learning. Given a gene name, a variation description, and supporting clinical text, the model predicts which of **9 mutation classes** the variant belongs to, directly mirroring the taxonomy used by Memorial Sloan Kettering Cancer Center (MSK).

---

## Dataset

**Source:** [MSK — Redefining Cancer Treatment (Kaggle)](https://www.kaggle.com/competitions/msk-redefining-cancer-treatment/overview)

| Split | Samples | Columns |
|---|---|---|
| Training | 3,321 | ID, Gene, Variation, Class (1–9), Text |
| Test | 5,668 | ID, Gene, Variation, Text |

**Files:**
- `training_variants` / `test_variants` — structured CSV with gene name, variation, and class label
- `training_text` / `test_text` — pipe-delimited (`||`) clinical literature excerpts per sample

**Class distribution (training set):**

| Class | Meaning | Approx. Samples |
|---|---|---|
| 1–9 | MSK-defined oncogenic mutation categories | Imbalanced across classes |

**Top genes by frequency:** BRCA1 (264), TP53 (163), EGFR (141), PTEN (126), BRCA2 (125)

---

## System Design

```
┌─────────────────────────────────────────────────────────────┐
│                        Raw Input Data                       │
│  training_variants.csv  ──────────┐                         │
│  training_text (||)    ──────────→│  pd.merge(on='ID')      │
└──────────────────────────────────┬──────────────────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │     Data Preprocessing      │
                    │  - Regex cleaning            │
                    │  - Lowercase + stopwords     │
                    │  - Punctuation removal       │
                    └──────────────┬──────────────┘
                                   │
          ┌────────────────────────┼────────────────────────┐
          │                        │                        │
          ▼                        ▼                        ▼
┌──────────────────┐   ┌──────────────────┐   ┌──────────────────┐
│   Text Encoder   │   │  Gene Encoder    │   │Variation Encoder │
│   Doc2Vec        │   │  One-Hot + SVD   │   │  One-Hot + SVD   │
│   → (N, 300)     │   │  → (N, 25)       │   │  → (N, 25)       │
└────────┬─────────┘   └────────┬─────────┘   └────────┬─────────┘
         │                      │                       │
         └──────────────────────┴───────────────────────┘
                                │
                    ┌───────────▼───────────┐
                    │   Feature Concat      │
                    │   (N, 350)            │
                    └───────────┬───────────┘
                                │
                    ┌───────────▼───────────┐
                    │  Keras Sequential NN  │
                    │  256→Drop→256→Drop    │
                    │      →80→9(softmax)   │
                    └───────────┬───────────┘
                                │
                    ┌───────────▼───────────┐
                    │  9-Class Prediction   │
                    │  (Mutation Category)  │
                    └───────────────────────┘
```

---

## Feature Engineering

The model uses three distinct feature modalities that are extracted in parallel and then concatenated:

### 1. Text Features — Doc2Vec (300 dimensions)

Clinical literature excerpts are embedded using Gensim's **Doc2Vec** (PV-DM variant):

| Hyperparameter | Value |
|---|---|
| `vector_size` | 300 |
| `window` | 5 |
| `min_count` | 1 |
| `sample` | 1e-4 |
| `negative` | 5 |
| `epochs` | 5 |
| `seed` | 1 |

The trained model is cached to `docEmbeddings.d2v` to avoid retraining on subsequent runs.

**Preprocessing pipeline applied before embedding:**
1. Regex filtering — keep only `[A-Za-z0-9^,!.\/'+-=]`
2. Lowercase conversion
3. NLTK English stopword removal
4. Punctuation stripping via `string.punctuation`

### 2. Gene Features — One-Hot Encoding + TruncatedSVD (25 dimensions)

Gene names are high-cardinality categoricals (hundreds of unique values). One-hot encoding followed by **TruncatedSVD** (`n_components=25, n_iter=25, random_state=12`) compresses this into a dense 25-dimensional representation while preserving co-occurrence variance.

### 3. Variation Features — One-Hot Encoding + TruncatedSVD (25 dimensions)

Same pipeline as genes. Mutation variations (e.g., `W802*`, `R1699Q`) are similarly encoded and reduced to 25 dimensions.

### Combined Feature Vector

```
Gene (25D)  +  Variation (25D)  +  Text (300D)  =  350D input to the neural network
```

The 350-dimensional vector for all 8,989 samples (train + test) is constructed with `np.hstack`.

**Target encoding:** Class labels (1–9) are integer-encoded via `LabelEncoder` then converted to one-hot via `to_categorical`, yielding a `(2656, 9)` training target matrix after the 80/20 split.

---

## Model Architecture

```
Input Layer:     (None, 350)
Dense(256, relu)         → 89,856 params
Dropout(0.3)
Dense(256, relu)         → 65,792 params
Dropout(0.5)
Dense(80,  relu)         → 20,560 params
Dense(9,   softmax)      →    729 params
────────────────────────────────────────
Total trainable params:   176,937
```

**Optimizer:** SGD with Nesterov momentum
```
learning_rate = 0.01
decay         = 1e-6
momentum      = 0.9
nesterov      = True
```

**Loss:** `categorical_crossentropy`  
**Training:** 50 epochs, batch size 32, 80/20 validation split

---

## Results

| Metric | Value |
|---|---|
| Avg. Training Accuracy (50 epochs) | 77.27% |
| Avg. Validation Accuracy (50 epochs) | 34.12% |
| Final Training Accuracy (epoch 50) | 87.84% |
| Final Validation Accuracy (epoch 50) | 34.89% |

Training accuracy improves steadily while validation accuracy plateaus around 34–36% — a clear sign of overfitting. The model learns the training distribution well but does not generalize, likely due to the small dataset size (2,656 training samples after split), class imbalance, and insufficient regularization for a 176K-parameter network.

---

## Architecture Decisions

### ADR-1: Doc2Vec for Text Representation
**Decision:** Use Gensim Doc2Vec rather than TF-IDF or bag-of-words.  
**Rationale:** Clinical mutation descriptions are long and semantically dense. TF-IDF treats terms independently; Doc2Vec learns distributed representations that capture semantic context (e.g., synonymous clinical terminology maps to nearby vectors). A 300D dense embedding also avoids the sparse high-dimensional input that would result from TF-IDF on medical vocabulary.

### ADR-2: TruncatedSVD at 25 Components for Categorical Features
**Decision:** One-hot encode genes and variations, then reduce with SVD to 25 dimensions.  
**Rationale:** Raw one-hot matrices are extremely sparse (hundreds of unique genes/variations). SVD identifies the principal axes of co-occurrence variance and produces a dense, compact representation. 25 components was chosen empirically as a balance between information retention and input dimensionality — keeping the combined feature vector at 350D rather than ~700D+.

### ADR-3: 300D Text vs. 25D Structural Features (Asymmetric Allocation)
**Decision:** Allocate 300 dimensions to text and only 25 each to gene and variation.  
**Rationale:** The clinical text is the primary discriminating signal — it contains the experimental evidence for class assignment. Gene and variation names are strong priors but lower-entropy signals (limited vocabulary, most classification information is in the text). The 6:1 ratio reflects this importance weighting.

### ADR-4: SGD with Nesterov Momentum over Adam
**Decision:** Use SGD (lr=0.01, momentum=0.9, Nesterov) instead of Adam.  
**Rationale:** Nesterov SGD tends to generalize better on imbalanced classification tasks by applying the momentum correction before computing the gradient, leading to more stable convergence. Adam's adaptive learning rates can sometimes overfit faster on small datasets.

### ADR-5: No Cross-Validation
**Decision:** Use a single 80/20 holdout split rather than k-fold cross-validation.  
**Rationale:** Notebook-based training with a 50-epoch Doc2Vec + neural network pipeline is computationally expensive per run. A single split was a pragmatic choice given course constraints. The downside is high variance in validation metrics.

---

## API Contracts

These define the expected data shapes and types at each stage boundary.

### Input Contract (raw sample)
```python
{
    "ID":        int,    # unique sample identifier
    "Gene":      str,    # e.g., "BRCA1", "TP53"
    "Variation": str,    # e.g., "R1699Q", "Truncating Mutations"
    "Text":      str     # clinical literature excerpt (may be None for 5 samples)
}
```

### Feature Vector Contract (model input)
```python
{
    "gene_features":      float[25],   # TruncatedSVD output from one-hot gene matrix
    "variation_features": float[25],   # TruncatedSVD output from one-hot variation matrix
    "text_features":      float[300],  # Doc2Vec document embedding
    "combined":           float[350]   # np.hstack([gene, variation, text])
}
```

### Model Output Contract
```python
{
    "probabilities": float[9],   # softmax output, sums to 1.0; index 0 → class 1
    "predicted_class": int,      # argmax(probabilities) + 1, range [1, 9]
    "confidence": float          # max(probabilities)
}
```

### Preprocessing Function Contracts
```python
def textClean(text: str) -> str:
    """Regex filter → lowercase → stopword removal. Returns cleaned string."""

def cleanup(text: str) -> str:
    """Calls textClean, then strips punctuation. Returns fully cleaned string."""

def constructLabeledSentences(data: pd.Series) -> list[TaggedDocument]:
    """Converts cleaned text Series to gensim TaggedDocument list for Doc2Vec training."""
```

---

## Setup & Usage

### Prerequisites

- Python 3.x
- Jupyter Notebook / JupyterLab

### Installation

```bash
git clone https://github.com/skushal746/CIS-6930-ML-in-Genomics.git
cd CIS-6930-ML-in-Genomics

sh setup_env.sh
python3 bring_dataset.py


```

### Data

Download the dataset from [Kaggle](https://www.kaggle.com/competitions/msk-redefining-cancer-treatment/data) and place the files in:

```
msk-redefining-cancer-treatment/
├── training_variants
├── training_text
├── test_variants
└── test_text
```

### Running

Open and run `CodeMLG.ipynb` top-to-bottom. On the first run, Doc2Vec training will execute and save `docEmbeddings.d2v`. Subsequent runs will load the cached model.

---

## Known Limitations

- **Overfitting:** Training accuracy (87.8%) far exceeds validation accuracy (34.1%). The model memorizes training samples rather than generalizing. Mitigation strategies to explore: stronger dropout, early stopping, L2 regularization, or data augmentation.
- **Small training set:** 2,656 effective training samples after the validation split is very small for a 9-class classification problem with high-dimensional text features.
- **Class imbalance:** No oversampling (SMOTE) or class-weighted loss is applied, meaning minority classes are likely under-predicted.
- **No test-set evaluation:** The notebook trains the model but does not generate predictions on the held-out test set or report per-class metrics (precision, recall, F1).
- **No early stopping:** Training runs for the full 50 epochs even after validation performance degrades — the best checkpoint is not saved.
