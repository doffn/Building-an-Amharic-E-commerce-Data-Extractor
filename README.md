
# 🛍️ EthioMart Amharic E-commerce NER Project

A 10 Academy AI Mastery Week 4 challenge to build a multilingual **Named Entity Recognition (NER)** system that extracts business-critical entities from Amharic Telegram messages. This project supports **EthioMart's vision** to centralize e-commerce activity and enable smart vendor evaluation for micro-lending.

---

## 📌 Project Summary

Telegram has become a powerful marketplace in Ethiopia. However, the lack of structure across vendor posts makes automation and analysis difficult. We aim to solve this by:
* Extracting entities like **Product Name**, **Price**, and **Location** from unstructured Amharic messages.
* Building a **Vendor Scorecard Engine** to help EthioMart assess business performance for **micro-lending** decisions.

---

## 📁 Project Structure

```

├── data/
│   ├── raw/                 \# Scraped Telegram posts (text, images)
│   ├── processed/           \# Cleaned and tokenized data
│   └── labelled/            \# CoNLL formatted labeled data
│
├── models/                  \# Fine-tuned model checkpoints
│
├── notebooks/               \# Jupyter notebooks (EDA, training, interpretability)
│
├── scripts/                 \# Python scripts for scraping, labeling, training, etc.
│
├── outputs/                 \# Evaluation reports, visualizations, vendor scorecards
│
├── .github/workflows/       \# GitHub Actions CI/CD workflows
│
├── .env                     \# Environment variables (API keys, etc.)
├── requirements.txt         \# Project dependencies
└── README.md                \# This file

````

---

## ✅ Key Tasks & Goals

### Task 1: Data Ingestion & Preprocessing
* Scrape messages from 5+ Telegram channels
* Extract text, images, timestamps, and metadata
* Preprocess Amharic text (tokenization, normalization)

### Task 2: NER Labeling (CoNLL Format)
* Annotate 30–50 messages with `Product`, `Price`, and `Location` tags
* Format using BIO tagging (`B-Product`, `I-Product`, etc.)

### Task 3: Fine-Tune NER Models
* Utilize Hugging Face models: `XLM-Roberta`, `bert-tiny-amharic`, or `afroxlmr`
* Fine-tune on the custom labeled dataset using the `transformers` library

### Task 4: Model Comparison & Selection
* Compare models based on F1-score, inference speed, and robustness
* Recommend the best model for potential production deployment

### Task 5: Model Interpretability
* Employ SHAP and LIME to explain model predictions
* Analyze failure cases and identify potential model biases

### Task 6: FinTech Vendor Scorecard Development
* Calculate key metrics:
    * Average views per post
    * Average posts per week
    * Average price per product
* Compute a simple **Lending Score** to effectively rank vendors for micro-lending decisions.

---

## 🚀 Setup & Installation

To get this project up and running locally, follow these steps:

```bash
# 1. Clone the repository
git clone [https://github.com/](https://github.com/)<your-username>/ethio-ner-project.git
cd ethio-ner-project

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate   # On Windows, use `venv\Scripts\activate`

# 3. Install project dependencies
pip install -r requirements.txt

# 4. Configure environment variables (e.g., API keys)
# Copy the example file and fill in your details.
cp .env.example .env
````

-----

## 🏃 How to Run (Example)

After setting up, you can run various scripts or notebooks. For example, to start data scraping:

```bash
# Example: To run the data scraping script
python scripts/scrape_telegram.py
# Or to run a notebook for training:
jupyter notebook notebooks/03_model_training.ipynb
```

*(Note: Adjust the example commands above to match your actual script names or typical workflow.)*

-----

## 🔑 Key Dependencies

  * `telethon`, `python-telegram-bot` – For Telegram data scraping
  * `transformers`, `datasets`, `torch` – Essential for model training and management
  * `seqeval` – For robust NER model evaluation metrics
  * `shap`, `lime` – For model interpretability and understanding predictions
  * `python-dotenv` – To manage environment configurations and API keys
  * `pandas`, `matplotlib`, `scikit-learn` – For data analysis, visualization, and general machine learning utilities

-----


