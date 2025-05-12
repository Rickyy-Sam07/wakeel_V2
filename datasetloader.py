import os
import faiss
import numpy as np
from flask import Flask, request, jsonify
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from groq import Groq

# --- Load Dataset ---
print("Loading Indian Law dataset...")
dataset = load_dataset("kshitij230/Indian-Law")

# --- Prepare Text Corpus ---
texts = [f"{item['Instruction']} {item['Response']}" for item in dataset['train']]

# --- TF-IDF Vectorization ---
print("Building vector index...")
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(texts)
tfidf_array = tfidf_matrix.toarray()

# --- FAISS Index ---
dimension = tfidf_array.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(tfidf_array)