# Automated Content Tagging for Blogs

**Persevex AI/ML Internship Project**

Repository: [https://github.com/kamalikasenthilnaathan09/automated-content-tagging](https://github.com/kamalikasenthilnaathan09/automated-content-tagging)

---

## 📌 Project Objective

The objective of this project is to build an end-to-end **multi-label text classification system** that accepts a blog title and article content, and automatically suggests **up to 5 relevant tags** constrained to a curated, approved tag taxonomy.

The system is designed to streamline editorial workflows, ensure consistency across digital publications, and power recommendation engines. It includes an interactive editor review loop where suggestions can be accepted, rejected, or refined.

---

## 🧠 Three Mandatory Machine Learning Approaches

To understand and benchmark different methodologies for multi-label text classification, three distinct models are implemented and evaluated:

1. **Model 1: Keyword Matching Baseline**
   - A rule-based lexicon baseline using the approved tag taxonomy, token matching, n-grams, and alias/synonym mappings.
   - Serves as the high-speed, interpretable benchmark.
2. **Model 2: TF-IDF + One-vs-Rest Logistic Regression**
   - Statistical NLP approach extracting n-gram Term Frequency-Inverse Document Frequency (TF-IDF) feature vectors.
   - Trains independent One-vs-Rest (OvR) binary logistic regression classifiers for each tag in the taxonomy with probability outputs.
3. **Model 3: Pretrained Sentence Transformer Embeddings + Multi-Label Head**
   - Dense semantic representations using a pretrained Sentence Transformer (e.g., `all-MiniLM-L6-v2`).
   - Implements text chunking with pooling to handle long blog articles exceeding token limits, coupled with a multi-label classification head.

> *Note: External generative LLM APIs, zero-shot models, KeyBERT, and BERTopic are deliberately excluded to ensure self-contained, reproducible, and verifiable ML implementations.*

---

## 🛠️ Technology Stack

- **Machine Learning & NLP**: Python, Pandas, NumPy, Scikit-learn, Sentence-Transformers
- **Text Extraction & Cleaning**: BeautifulSoup4, lxml
- **Serialization & Storage**: Joblib, SQLAlchemy, SQLite
- **Backend API**: FastAPI, Uvicorn
- **Frontend**: React (interactive editor tagging UI)
- **Evaluation & Visualization**: Matplotlib, Seaborn, Jupyter

---

## 📁 Project Folder Structure

```text
Automated content tagging/
├── README.md                  # Project overview, approaches, and structure
├── requirements.txt           # Python library dependencies
├── .gitignore                 # Files excluded from git tracking
├── data/
│   ├── raw/                   # Original downloaded datasets (unprocessed)
│   ├── processed/             # Cleaned and normalized datasets
│   └── splits/                # Train, validation, and test splits
├── notebooks/                 # Jupyter notebooks for EDA and step-by-step experimentation
├── src/
│   ├── data/                  # Dataset ingestion and loading scripts
│   ├── preprocessing/         # Text cleaning, HTML stripping, and tag binarization
│   ├── taxonomy/              # Approved tag taxonomy, alias mappings, and validation
│   ├── models/                # Implementations of Model 1, Model 2, and Model 3
│   └── evaluation/            # Evaluation metrics (Micro/Macro F1, P@3, R@3, latency)
├── backend/                   # FastAPI application & SQLite database for editor workflow
├── frontend/                  # React web application for editors
└── reports/                   # Performance benchmarks, evaluation tables, and figures
```

---

## 📊 Evaluation Criteria

All models will be compared on holdout test data using:
- **Micro-F1** and **Macro-F1**
- **Precision@3** and **Recall@3**
- **Inference Latency** (ms per article)
- **Frequent vs. Rare Tag Performance**
- **Error Analysis** (false positive suggestions and missed tags)
