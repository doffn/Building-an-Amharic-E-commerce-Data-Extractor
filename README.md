
```markdown
# 🛍️ EthioMart Amharic E-commerce NER Project

A 10 Academy AI Mastery Week 4 challenge to build a multilingual **Named Entity Recognition (NER)** system that extracts business-critical entities from Amharic Telegram messages. This project supports **EthioMart's vision** to centralize e-commerce activity and enable smart vendor evaluation for micro-lending.

---

## 📌 Project Summary

Telegram has become a powerful marketplace in Ethiopia. However, the lack of structure across vendor posts makes automation and analysis difficult. We aim to solve this by:
- Extracting entities like **Product Name**, **Price**, and **Location** from unstructured Amharic messages.
- Building a **Vendor Scorecard Engine** to help EthioMart assess business performance for **micro-lending** decisions.

---

## 📁 Project Structure

```

├── data/
│   ├── raw/               # Scraped Telegram posts (text, images)
│   ├── processed/         # Cleaned and tokenized data
│   └── labelled/          # CoNLL formatted labeled data
│
├── models/                # Fine-tuned model checkpoints
│
├── notebooks/             # Jupyter notebooks (EDA, training, interpretability)
│
├── scripts/               # Python scripts for scraping, labeling, training, etc.
│
├── outputs/               # Evaluation reports, visualizations, vendor scorecards
│
├── .github/workflows/     # GitHub Actions CI/CD workflows
│
├── .env                   # Environment variables (API keys, etc.)
├── requirements.txt       # Project dependencies
└── README.md              # This file

````

---

## ✅ Tasks & Goals

### Task 1: Data Ingestion & Preprocessing
- Scrape messages from 5+ Telegram channels
- Extract text, images, timestamps, and metadata
- Preprocess Amharic text (tokenization, normalization)

### Task 2: NER Labeling (CoNLL Format)
- Annotate 30–50 messages with `Product`, `Price`, and `Location` tags
- Format using BIO tagging (`B-Product`, `I-Product`, etc.)

### Task 3: Fine-Tune NER Models
- Use Hugging Face models: `XLM-Roberta`, `bert-tiny-amharic`, or `afroxlmr`
- Fine-tune on labeled dataset using the `transformers` library

### Task 4: Model Comparison
- Compare models on F1-score, inference speed, and robustness
- Recommend the best model for production deployment

### Task 5: Interpretability
- Use SHAP and LIME to explain model predictions
- Analyze failure cases and model biases

### Task 6: FinTech Vendor Scorecard
- Calculate: 
  - Avg. views/post
  - Avg. posts/week
  - Avg. price per product
- Compute a simple **Lending Score** to rank vendors

---

## 📦 Setup & Installation

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/ethio-ner-project.git
cd ethio-ner-project

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your API keys in .env
cp .env.example .env
````

---

## 🔑 Key Dependencies

* `telethon`, `python-telegram-bot` – Telegram scraping
* `transformers`, `datasets`, `torch` – Model training
* `seqeval` – NER evaluation
* `shap`, `lime` – Interpretability
* `dotenv` – Environment config
* `pandas`, `matplotlib`, `scikit-learn` – Analytics & plotting

---

