
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
│   ├── raw/                 \# Scraped Telegram posts (text, images)
|       ├── channels.txt      \# Channels used for scrapping
|       ├── labeled\_data\_from\_df.conll   \# samples for labeled data
|       ├── telegram\_data.csv            \# Scrapped Datas
│
├── models/               \# Ideally include the models ( too large)
├── notebooks/               \# Jupyter notebooks (EDA, training, interpretability)
│   ├── task1\_2.ipynb        \# Notebook for task 1 and 2
│   ├── task3\_4ipynb.ipynb   \# Notebook for fine-tuning and model comparison
│
├── photos/                  \# Photos scraped
├── scripts/                 \# Python scripts for scraping, labeling, training, etc.
│   ├── telegram\_scraper.py   \# Script to extract message from telegram channels
│
├── .github/workflows/       \# GitHub Actions CI/CD workflows
│
├── .env                     \# Environment variables (API keys, etc.)
├── requirements.txt         \# Project dependencies
└── README.md                \# This file

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

## 💡 Model Fine-tuning & Evaluation Details

The `notebooks/task3_4ipynb.ipynb` notebook outlines the process for fine-tuning and evaluating the Named Entity Recognition (NER) models.

The general workflow involves:
1.  **Loading Pre-trained Models and Tokenizers**: Models and their corresponding tokenizers, such as `Davlan/bert-tiny-ner-amharic`, are loaded from the Hugging Face Hub. This includes downloading necessary configuration files, tokenizer models, and model weights (e.g., `model.safetensors`).
2.  **Data Preparation**: The labeled dataset is prepared for model training. This crucial step involves tokenizing the text and aligning the NER labels with the new tokenized input, ensuring that the model correctly understands which tokens correspond to which entities.
3.  **Model Training**: The pre-trained models are fine-tuned on the custom Amharic NER dataset. This process leverages the `transformers` library to adapt the models' understanding to the specific entities (`Product`, `Price`, `Location`) relevant to Ethiopian e-commerce Telegram messages.
4.  **Model Evaluation**: After training, the models are evaluated using metrics such as F1-score, precision, and recall to assess their performance on the NER task. This step helps in comparing different models and selecting the most effective one for deployment.

*(Note: The provided notebook content (`task3_4ipynb.ipynb`) was partial and primarily showed progress bar outputs for model and tokenizer downloads and dataset mapping. Therefore, detailed Python code snippets for the entire fine-tuning and evaluation process could not be extracted directly.)*

* Model can be accessed using this URL: [LINK 🔗]([https://huggingface.co/doffn/tenx_best_NLP_model](https://huggingface.co/doffn/tenx_best_NLP_model/tree/main))

---

## 💻 Hugging Face Integration

Hugging Face is a leading platform and community for machine learning, particularly known for its open-source libraries like `transformers`, which makes state-of-the-art pre-trained models readily accessible for various natural language processing (NLP) tasks.

In the context of this project, Hugging Face plays a crucial role in:

* **Providing Pre-trained Models**: The project leverages pre-trained transformer models from the Hugging Face Hub, such as `XLM-Roberta`, `bert-tiny-amharic`, and `afroxlmr`. These models have been trained on vast amounts of text data, allowing them to understand language context and nuances, which is a powerful starting point for our specific task. The `notebooks/task3_4ipynb.ipynb` notebook specifically shows the loading of a `Davlan/bert-tiny-ner-amharic` model and its tokenizer, indicating its use in the fine-tuning process.
* **Facilitating Fine-tuning**: The Hugging Face `transformers` library provides a robust framework and tools (like `Trainer` API) to efficiently fine-tune these pre-trained models on our custom, smaller Amharic e-commerce NER dataset. This adaptation makes the general-purpose models highly effective for identifying domain-specific entities.

### Output of the Models

The models built in this project are **Named Entity Recognition (NER)** models. Their primary output, after processing an input Amharic Telegram message, is a sequence of **BIO tags** (Beginning, Inside, Outside) corresponding to each word or token in the input. These tags indicate the type of entity and its position within the entity span:

* `B-Product`: Beginning of a product name.
* `I-Product`: Inside a product name.
* `B-Price`: Beginning of a price.
* `I-Price`: Inside a price.
* `B-Location`: Beginning of a location.
* `I-Location`: Inside a location.
* `O`: Outside of any defined entity.

From these BIO tags, the model can then extract the complete entities. For example, if the model outputs `B-Product` for "የ", `I-Product` for "ፀጉር", and `I-Product` for "ዘይት", the extracted entity would be "የፀጉር ዘይት" (hair oil).

### Types of Models Built

The project focuses on fine-tuning **transformer-based models** for the Named Entity Recognition task. The specific types of models utilized or considered for fine-tuning include:

* **`XLM-Roberta`**: A multilingual language model capable of understanding and generating text in multiple languages, including Amharic, making it suitable for cross-lingual tasks or languages with fewer resources.
* **`bert-tiny-amharic` (specifically `Davlan/bert-tiny-ner-amharic`)**: A BERT-based model specifically pre-trained or fine-tuned for the Amharic language. The `tiny` version indicates a smaller model size, which can be beneficial for faster inference and deployment in resource-constrained environments.
* **`afroxlmr`**: Another multilingual transformer model from the African NLP community, often used for African languages, which could provide strong performance for Amharic.

These base models are then **fine-tuned** on the project's custom-labeled Amharic e-commerce dataset to specialize them in identifying `Product`, `Price`, and `Location` entities. The "types of models built" are thus these pre-trained language models adapted for the specific NER task.

---

## 🚀 Setup & Installation

To get this project up and running locally, follow these steps:

```bash
# 1. Clone the repository
git clone [https://github.com/](https://github.com/)<your-username>/ethio-ner-project.git
cd ethio-ner-project

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate   # On Windows, use `venv\Scripts\activate`

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
# To run the notebook for fine-tuning and evaluation:
jupyter notebook notebooks/task3_4ipynb.ipynb
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


