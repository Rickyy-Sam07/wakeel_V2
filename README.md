
# 🧑‍⚖️ Wakeel Sahab – Indian Law RAG Assistant

**Wakeel Sahab** is a Retrieval-Augmented Generation (RAG) based AI assistant designed to help users navigate Indian law. It classifies queries, retrieves relevant legal documents, and generates context-aware answers in a structured, point-wise manner.

---

## 📁 Project Structure

```
├── data_loader.py   # Loads dataset and builds vector index using TF-IDF + FAISS
├── main.py          # Runs the chat pipeline with classification, RAG, and response generation
├── requirements.txt # Dependencies
```

---

## 🚀 Getting Started

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Make sure to include your `Groq` API key in the appropriate section of the code.

---

### 2. Load the Dataset and Build the Index

Before running the main app, load and prepare the dataset:

```bash
python data_loader.py
```

This script:

* Loads the Indian Law dataset
* Vectorizes documents using TF-IDF
* Builds a FAISS index for fast retrieval

---

### 3. Start the Chat Assistant

Once the index is built, you can start the chatbot:

```bash
python main.py
```

Features:

* Classifies queries as **inner (legal)** or **outer (general/greeting)**
* Retrieves top-K relevant legal documents for legal queries
* Uses LLaMA 3 via Groq for final, structured answers

You can use the **`special keys`** keyword for admin-level debugging and full RAG output preview.

---

## ⚖️ Why Wakeel Sahab?

* Built specifically for Indian legal queries
* Fast and accurate retrieval using FAISS
* Friendly interaction + structured legal guidance

---

## 📌 Coming soon ....

* Add a web interface (Flask or Streamlit)
* Logging & analytics
* Expand dataset coverage

---

## 🙌 Contributions

Ideas, feedback, or improvements? Feel free to open issues or pull requests!
