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


    def evaluate(self, reference, generated):
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
        return scorer.score(reference, generated)

    def compression_ratio(self, original, summary):
        original_words = max(1, len(word_tokenize(str(original))))
        summary_words = len(word_tokenize(str(summary)))
        return summary_words / original_words

    def important_keywords(self, text, top_k=15):
        """Top TF-IDF terms from the full document (for display / 'main ideas' hints)."""
        processed = self.preprocess(str(text))
        if not processed.strip():
            return []
        vec = TfidfVectorizer(stop_words="english", max_features=5000)
        matrix = vec.fit_transform([processed])
        terms = vec.get_feature_names_out()
        scores = matrix.toarray().flatten()
        if len(scores) == 0:
            return []
        k = min(top_k, len(scores))
        top_idx = np.argsort(scores)[-k:][::-1]
        return [(str(terms[i]), float(scores[i])) for i in top_idx]

    def keyword_recall(self, original, summary, top_k=15):
        original_processed = self.preprocess(str(original))
        summary_processed = self.preprocess(str(summary))
        if not original_processed.strip():
            return 0.0

        vec = TfidfVectorizer(stop_words='english')
        matrix = vec.fit_transform([original_processed])
        terms = vec.get_feature_names_out()
        scores = matrix.toarray().flatten()

        if len(scores) == 0:
            return 0.0

        top_idx = np.argsort(scores)[-min(top_k, len(scores)):]
        important_keywords = set(terms[top_idx])
        summary_tokens = set(word_tokenize(summary_processed))

        if not important_keywords:
            return 0.0
        return len(important_keywords & summary_tokens) / len(important_keywords)

   def sentence_selection_classification_metrics(self, scores, y_true, k):
        """
        Treat top-k highest-scoring sentences as predicted 'important' (1) vs not (0).
        """
        scores = np.asarray(scores, dtype=float)
        y_true = np.asarray(y_true, dtype=int)
        if scores.size == 0 or y_true.size == 0 or scores.shape != y_true.shape:
            return None
        k = max(1, min(int(k), len(scores)))
        y_pred = np.zeros_like(y_true)
        top_idx = np.argsort(scores)[-k:]
        y_pred[top_idx] = 1
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
            "y_true": y_true,
            "y_pred": y_pred,
        }

   def _minmax_normalize(self, arr):
        arr = np.array(arr, dtype=float)
        min_val = float(np.min(arr))
        max_val = float(np.max(arr))
        if max_val - min_val < 1e-12:
            return np.ones_like(arr)
        return (arr - min_val) / (max_val - min_val)
       
   def key_sentence_ranking(self, text, n=3, alpha=0.6):
        """Human-readable ranking: TF-IDF-only and hybrid scores per sentence."""
        sentences_t, tfidf_s = self._tfidf_sentence_scores(text)
        sentences_h, hybrid_s = self._hybrid_sentence_scores(text, alpha=alpha)
        out = {"sentences": sentences_t, "tfidf_scores": tfidf_s, "hybrid_scores": hybrid_s}
        if len(sentences_t) == 0:
            out["top_tfidf"] = []
            out["top_hybrid"] = []
            return out
        order_t = np.argsort(tfidf_s)[::-1][: min(n, len(sentences_t))]
        order_h = np.argsort(hybrid_s)[::-1][: min(n, len(sentences_h))]
        out["top_tfidf"] = [(int(i), sentences_t[i], float(tfidf_s[i])) for i in order_t]
        out["top_hybrid"] = [(int(i), sentences_h[i], float(hybrid_s[i])) for i in order_h]
        return out

