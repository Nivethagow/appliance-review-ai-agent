# 🏠 AI Agent for Major Appliance Sellers

An AI-powered NLP pipeline that extracts structured, actionable insights from unstructured Amazon customer reviews of major appliances — combining sentiment classification and topic modeling to help sellers understand what customers love and what needs improvement.

---

## 📌 Problem Statement

Customer reviews are abundant but rarely analyzed systematically. Marketing teams need concrete, actionable suggestions — not just a "positive/negative" label. This project addresses that gap by building an AI agent that:

- Automatically classifies review sentiment
- Extracts key themes from positive and negative reviews separately
- Outputs structured, per-review insights that can drive product and marketing decisions

---

## 🗂️ Dataset

**Source:** [Kaggle — Amazon US Customer Reviews Dataset](https://www.kaggle.com/datasets/cynthiarempel/amazon-us-customer-reviews-dataset)

- ~97,000 major appliance reviews (2014–2021)
- Key fields used: `review_body`, `star_rating`, `review_id`
- Preprocessing steps: HTML stripping, lowercasing, punctuation removal, stopword filtering, lemmatization

---

## 🏗️ Architecture

```
Data Preprocessing
        ↓
Sentiment Analysis (TextBlob)
        ↓
Split → Positive Reviews | Negative Reviews
        ↓                        ↓
  LDA Topic Model          LDA Topic Model
  (Positive Corpus)        (Negative Corpus)
        ↓                        ↓
     C_V Coherence Score Evaluation
               ↓
      Orchestrator Pipeline
               ↓
    Output: { review_id, sentiment, key_insights }
```

### Agents

| Agent | Role |
|---|---|
| **Sentiment Agent** | Uses TextBlob polarity scoring to classify each review as positive (≥ 0) or negative (< 0) |
| **Key-Insights Agent** | Trains two separate LDA models (positive corpus / negative corpus) to extract top thematic keywords per review |
| **Orchestrator** | Coordinates the pipeline: preprocesses → classifies → routes → extracts → outputs structured JSON-style records |

The dual-pipeline LDA design prevents positive reviews (83k) from drowning out negative ones (13k), ensuring more balanced and interpretable topic extraction.

---

## 📊 Results

| Metric | Value |
|---|---|
| Positive reviews | 83,159 |
| Negative reviews | 13,668 |
| Positive LDA Coherence (C_V) | 0.5935 |
| Negative LDA Coherence (C_V) | 0.5915 |

**Positive review themes:** work, fit, great, product, price, water, service, fridge, temperature, dishwasher

**Negative review themes:** product, unit, water, work, service, warranty, cold, oven, washer, refrigerator

**Key finding:** Overlapping generic keywords (e.g., "unit", "product") across both sentiments highlights the importance of contextual interpretation beyond simple keyword matching.

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install pandas numpy matplotlib nltk textblob scikit-learn gensim ace-tools-open
```

```python
import nltk
nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger_eng')
```

### Running the Notebook

1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/cynthiarempel/amazon-us-customer-reviews-dataset) and place the `.tsv` file in your working directory (or Google Drive if using Colab).
2. Update the file path in the data loading cell:
   ```python
   df = pd.read_csv('path/to/amazon_major_appliances.tsv', sep='\t', on_bad_lines='skip')
   ```
3. Run all cells in `notebooks/appliance_review_ai_agent.ipynb`.

### Output Format

Each row in the final output DataFrame:

```
review_id        sentiment    key_insights
R203HPW78Z7N4K   positive     [oven, range, microwave, look, hood]
R2EAIGVLEALSP3   positive     [great, work, fit, product, price]
R29AX0PMYDL4FP   negative     [ice, maker, product, time, work]
```

---

## 📁 Project Structure

```
appliance-review-ai-agent/
│
├── notebooks/
│   └── appliance_review_ai_agent.ipynb   # Full pipeline notebook
│
├── outputs/                              # Exported CSV results
│
├── assets/                               # Charts and visualizations
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## ⚠️ Limitations

- **Class imbalance:** ~86% positive reviews limits negative topic diversity
- **Classical NLP only:** LLMs (e.g., GPT) were excluded; TextBlob polarity can miss sarcasm/irony
- **Keyword overlap:** Generic terms appear in both sentiment buckets, reducing interpretability

---

## 🔭 Future Work

- **LLM Integration** — Use Claude or GPT for deeper, context-aware insight extraction
- **Real-Time Analytics** — Stream incoming reviews through the pipeline for live dashboards
- **Multimodal Feedback** — Incorporate star ratings, images, and video reviews
- **Cross-Domain Application** — Extend to automotive, beauty, restaurants, electronics

---

## 👤 Author

**Nivetha Ramasamy**

---

## 📄 License

This project is for academic and research purposes.
