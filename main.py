import pandas as pd
import numpy as np
import string
import os
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sentence_transformers import SentenceTransformer
from transformers import BartForConditionalGeneration, BartTokenizer
from rouge_score import rouge_scorer
import torch
import re

def _ensure_nltk():
    for path, name in (
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("corpora/stopwords", "stopwords"),
    ):
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(name, quiet=True)

_ensure_nltk()


class Summarizer:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.tfidf = TfidfVectorizer(stop_words='english')
        
    
        print("Status: Loading Embedding Model...")
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')

       
        print("Status: Loading BART Model (this may take a minute)...")
        self.tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
        self.model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')


    def preprocess(self, text):
         text = text.lower() 
         text = text.translate(str.maketrans('', '', string.punctuation)) # Remove punctuation
         tokens = word_tokenize(text)
         tokens = [w for w in tokens if w not in self.stop_words] # Remove stopwords
         return " ".join(tokens)
            

    def _tfidf_sentence_scores(self, text):
         sentences = sent_tokenize(text)
         if not sentences:
             return sentences, np.array([], dtype=float)
         processed = [self.preprocess(s) for s in sentences]
         matrix = self.tfidf.fit_transform(processed).toarray()
         scores = cosine_similarity(matrix).sum(axis=1)
         return sentences, np.asarray(scores, dtype=float)

    def tfidf_summary(self, text, n=3):
         sentences, scores = self._tfidf_sentence_scores(text)
         if len(sentences) <= n:
             return text
         top_indices = np.argsort(scores)[-n:]
         top_indices = sorted(top_indices)
         return " ".join([sentences[i] for i in top_indices])

    def summarize_hybrid(self, text):
        pass

    def bart_summary(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
        
        summary_ids = self.model.generate(
            inputs["input_ids"],
            max_length=130, 
            min_length=40, 
            length_penalty=2.0,
            num_beams=4, 
            early_stopping=True
        )
        return self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)


    def evaluate(self, hypothesis, reference):
        pass
