import os
import faiss
import numpy as np
from flask import Flask, request, jsonify
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from groq import Groq
from datasetloader import *

client = Groq(api_key='api_key_here')

def retrieve_documents(query, top_k=5):
    query_vec = vectorizer.transform([query]).toarray()
    distances, indices = index.search(query_vec, top_k)
    return [texts[i] for i in indices[0]]

# --- Query Classifier (inner/outer) ---
def classify_query(query):
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": "Classify this query. Return 'inner' if it's a legal question under Indian law or law case based question where legal actions is neede, else return 'outer'."
                },
                {
                    "role": "user",
                    "content": query
                }
            ],
            max_tokens=5
        )
        return response.choices[0].message.content.strip().lower()
    except Exception as e:
        print("Classification error:", e)
        return "outer"

# --- Greeting Bot (outer) ---
def greeting_bot(query):
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "Respond like a polite Indian legal assistant. Handle greetings, small talk, or user emotions."},
                {"role": "user", "content": query}
            ],
            max_tokens=100
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {e}"

# --- RAG with Groq (inner) ---
def generate_legal_response(query, context):
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": "You are a legal assistant specializing in Indian law. Use context to generate a helpful answer with relevant laws or articles. Don't go beyond Indian law. give answer in points and in structures way"
                },
                {
                    "role": "user",
                    "content": f"Query: {query}\n\nContext:\n{context}"
                }
            ],
            max_tokens=500
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {e}"

# --- Final RAG Pipeline ---
def complete_query(query):
    classification = classify_query(query)
    print("Classification:", classification)

    if classification == 'outer':
        return greeting_bot(query)

    # Otherwise, legal (inner)
    context = "\n".join(retrieve_documents(query))
    return generate_legal_response(query, context)

def rag_with_groq(query):
    context = "\n".join(retrieve_documents(query))
    return generate_legal_response(query, context)

def colab_chat():
    print("👨‍⚖️ Wakeel Sahab (Legal RAG Bot) is ready! Type 'q' to quit.\n")
    while True:
        query = input("Ok tell me what happened: ").strip()

        if query.lower() == 'q':
            print("👋 Exiting... Thank you!")
            break

        if ".god" in query.lower():
            print("🔍 Key '.god' detected — Forcing full RAG pipeline.")
            context = "\n".join(retrieve_documents(query))
            r= retrieve_documents(query)
            print("rag fetched data is :",r)
            response = generate_legal_response(query, context)
            print("Mr. Wakeel Sahab:\n", response)
            print("-" * 50)
            continue

        classification = classify_query(query).strip().lower()
        print("Classification:", classification)

        if classification == 'outer':
            response = greeting_bot(query)
        else:
            context = "\n".join(retrieve_documents(query))
            response = response = rag_with_groq(query)


        print("Mr. Wakeel Sahab:\n", response)
        print("-" * 50)

# Call this to start the chat in Colab
colab_chat()
 