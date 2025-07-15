# End-to-End ML Pipeline

This project builds a **production-ready machine learning pipeline** using `scikit-learn` to predict customer churn from the Telco dataset.

## Features

- Data preprocessing using `ColumnTransformer`
- Supports both **Logistic Regression** and **Random Forest**
- Hyperparameter tuning with `GridSearchCV`
- Model evaluation using `classification_report`
- Final trained pipeline saved with `joblib`

## Workflow

1. Drops `customerID` and cleans `TotalCharges` column
2. Splits numeric and categorical features
3. Applies scaling and one-hot encoding
4. Trains models with cross-validation
5. Evaluates the best model
6. Saves the pipeline to disk as `telco_churn_pipeline.joblib`

## Model Usage

```python
import joblib
model = joblib.load("telco_churn_pipeline.joblib")
predictions = model.predict(new_data)
```

## Dependencies

* pandas
* numpy
* scikit-learn
* joblib
----------------
----------------
   

# News Headline Classifier (BERT + Streamlit)

This project fine-tunes a **BERT model** (`bert-base-uncased`) on the [AG News dataset](https://huggingface.co/datasets/ag_news) to classify news headlines into 4 categories: **World, Sports, Business, Sci/Tech**. The model is then deployed using **Streamlit**.

---

## Features

- Fine-tunes a BERT model using HuggingFace `transformers`
- Evaluates with accuracy and F1 score
- Saves the trained model and tokenizer
- Deploys with a simple Streamlit UI
- Accessible via public URL using `pyngrok` (Colab-friendly)

---

## Dependencies

Install all requirements via pip:

```bash
pip install transformers datasets scikit-learn streamlit pyngrok
```
---------
---------

#  Context-Aware Chatbot using LangChain + HuggingFace + FAISS

This project builds a **retrieval-augmented chatbot** using LangChain, a custom text corpus (`corpus.txt`), and a lightweight HuggingFace model. It supports **contextual memory** and **document-based Q&A** via vector search.

---

## Features

- Uses **LangChain's ConversationalRetrievalChain**
- Embeds and stores `corpus.txt` using **FAISS** and **HuggingFace embeddings**
- Remembers conversation context using memory
- Generates responses with a **lightweight Hugging Face model** (e.g., `LaMini-Flan-T5`)
- Runs fully locally — no OpenAI API key required

---

## Setup & Run

### 1. Install Dependencies

```bash
pip install langchain faiss-cpu transformers sentence-transformers
```