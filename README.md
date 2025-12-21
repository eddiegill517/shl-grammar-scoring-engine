# Grammar Scoring Engine for Spoken Audio

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![Whisper](https://img.shields.io/badge/OpenAI-Whisper-green.svg)](https://github.com/openai/whisper)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-orange.svg)](https://huggingface.co)

**End-to-end ML pipeline for automated grammar assessment of spoken English**


# 📌 Problem Statement

Build a model that takes **audio files (45-60 sec)** as input and outputs a **continuous grammar score (1-5)** based on MOS Likert Grammar Scoring rubric.

| Dataset | Samples |
|---------|---------|
| Training|  409    |
| Testing |  197    |


# 🧠 Methodology

## Pipeline Overview
Audio (.wav) → Whisper Transcription → Multi-Modal Feature Extraction → Ensemble Model → Score [1-5]

## Feature Extraction (477 total features)

| Component               | Model/Method                                      | Features |
|-------------------------|---------------------------------------------------|----------|
| **Grammar Scoring**     | RoBERTa fine-tuned on CoLA                        | 3   |
| **Semantic Embeddings** | DeBERTa-v3-small                                  | 384 |
| **Linguistic Features** | Handcrafted (readability, complexity, vocabulary) | 50  |
| **Audio Features**      | Librosa (MFCCs, pitch, pauses, speech rate)       | 40  |

## Models Used

**12-Model Ensemble with Optimized Weights:**
- XGBoost (2 variants)
- LightGBM (2 variants)  
- Gradient Boosting
- Random Forest
- Extra Trees
- Ridge Regression
- Bayesian Ridge
- Huber Regressor
- SVR
- KNN Regressor

**Optimization:** Nelder-Mead algorithm minimizing OOF RMSE with 5-Fold CV

---

## 📈 Results

### Model Performance (Out-of-Fold Cross-Validation)

| Metric | Score |
|--------|-------|
| **RMSE** | 0.5403 |
| **Pearson Correlation** | 0.7114 |
| **Spearman Correlation** | 0.6026 |

### Individual Model Performance

| Model | RMSE | Pearson r |
|-------|------|-----------|
| SVR | 0.5524 | 0.6947 |
| Bayesian Ridge | 0.5531 | 0.6919 |
| XGBoost v2 | 0.5596 | 0.6840 |
| XGBoost v1 | 0.5624 | 0.6790 |
| LightGBM v1 | 0.5660 | 0.6738 |
| **Ensemble** | **0.5403** | **0.7114** |

### Optimized Ensemble Weights

| Model | Weight |
|-------|--------|
| Bayesian Ridge | 30.5% |
| KNN | 15.4% |
| LightGBM v1 | 14.3% |
| SVR | 13.1% |
| XGBoost v1 | 11.2% |
| Ridge | 10.9% |
| XGBoost v2 | 4.6% |

---

## 🔬 Key Technical Details

### Speech-to-Text
- **Model:** OpenAI Whisper (base)
- **Output:** English transcription of spoken audio

### Grammatical Acceptability  
- **Model:** `textattack/roberta-base-CoLA`
- **Purpose:** Sentence-level grammar probability scoring

### Semantic Understanding
- **Model:** `microsoft/deberta-v3-small`
- **Method:** CLS + Mean pooling → 384-dim embeddings

### Linguistic Analysis
- Lexical: word count, vocabulary richness, unique word ratio
- Syntactic: sentence length, subordinate clauses, complexity
- Readability: Flesch-Kincaid, Flesch Reading Ease scores
- Fluency: filler words (um, uh), contractions, discourse markers

### Audio Analysis
- MFCCs (13 coefficients)
- Speech rate & tempo
- Pause frequency & silence ratio
- Pitch statistics (F0 mean, std, range)


## 📊 Grammar Score Rubric

| Score | Description |
|-------|-------------|
| **1** | Struggles with sentence structure; limited grammatical control |
| **2** | Limited understanding; consistent basic mistakes; incomplete sentences |
| **3** | Decent structure but grammar errors, or decent grammar but syntax errors |
| **4** | Strong understanding; good control; minor errors don't affect comprehension |
| **5** | High accuracy; complex grammar control; effective self-correction |


## 🛠 Tech Stack
numpy, pandas, scikit-learn, torch, transformers
openai-whisper, librosa, xgboost, lightgbm
matplotlib, scipy, tqdm


## 📁 Files

| File                            | Description |
|---------------------------------|-------------|
| `SHLGrammarScoringEngine.ipynb` | Complete pipeline with training, evaluation, and predictions |
| `README.md`                     | Project documentation |


## 📚 References

- Radford et al. (2022) - *Whisper: Robust Speech Recognition*
- Warstadt et al. (2019) - *CoLA: Corpus of Linguistic Acceptability*
- He et al. (2021) - *DeBERTa: Decoding-enhanced BERT*
- Chen & Guestrin (2016) - *XGBoost*

## Built for SHL AI Team | Intern Hiring Assessment 2025
⭐ Star this repo if you found it interesting!
