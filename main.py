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
   def gold_sentence_labels_from_reference(self, sentences, reference, rouge1_f1_threshold=0.12):
        """Weak labels: sentence is 'important' if ROUGE-1 F1 vs reference summary is high enough.Used only to report Accuracy / Precision / Recall / F1 / Confusion matrix for extractive selection."""
        if not sentences or not str(reference).strip():
            return None
        scorer = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)
        labels = []
        for s in sentences:
            f1 = scorer.score(str(reference), s)["rouge1"].fmeasure
            labels.append(1 if f1 >= rouge1_f1_threshold else 0)
        return np.array(labels, dtype=int)
def run():
    file_path = "data.csv"

    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found in {os.getcwd()}")
        return

    df = pd.read_csv(file_path)
    print(f"Dataset loaded! Total Articles: {len(df)}")

    engine = Summarizer()
    aggregate = {
        "tfidf_rougeL": [],
        "hybrid_rougeL": [],
        "bart_rougeL": [],
        "tfidf_ratio": [],
        "hybrid_ratio": [],
        "bart_ratio": [],
        "tfidf_key": [],
        "hybrid_key": [],
        "bart_key": [],
        "tfidf_manual": [],
        "hybrid_manual": [],
        "bart_manual": [],
    }

    for i in range(min(3, len(df))):
        article_text = df['article'][i]
        reference_text = df['highlights'][i]

        print(f"\n" + "="*60)
        print(f"Analyzing Article #{i+1}")
        print("="*60)
        
        # Generate Summaries
        tfidf_res = engine.tfidf_summary(article_text)
        hybrid_res = engine.hybrid_extractive_summary(article_text)
        bart_res = engine.bart_summary(article_text)

        print(f"\n[Original Article]:\n{article_text[:600]}...")
        print(f"\n[Baseline Summary - TF-IDF]:\n{tfidf_res}")
        print(f"\n[Improved Extractive Summary - TF-IDF + Embeddings]:\n{hybrid_res}")
        print(f"\n[Advanced Summary - BART]:\n{bart_res}")

        keywords = engine.important_keywords(article_text, top_k=12)
        print("\n[Important keywords (TF-IDF over full article)]:")
        print(", ".join(f"{w} ({s:.3f})" for w, s in keywords[:12]) if keywords else "(none)")

        ranking = engine.key_sentence_ranking(article_text, n=3)
        sents = ranking["sentences"]
        tfidf_sc = ranking["tfidf_scores"]
        hybrid_sc = ranking["hybrid_scores"]
        print("\n[Top sentences by TF-IDF score]:")
        for idx, sent, sc in ranking.get("top_tfidf", []):
            short = (sent[:120] + "…") if len(sent) > 120 else sent
            print(f"  #{idx} score={sc:.4f} — {short}")
        print("\n[Top sentences by hybrid TF-IDF + embedding score]:")
        for idx, sent, sc in ranking.get("top_hybrid", []):
            short = (sent[:120] + "…") if len(sent) > 120 else sent
            print(f"  #{idx} score={sc:.4f} — {short}")

        y_gold = engine.gold_sentence_labels_from_reference(sents, reference_text)
        if y_gold is not None and y_gold.sum() == 0:
            y_gold = engine.gold_sentence_labels_from_reference(
                sents, reference_text, rouge1_f1_threshold=0.06
            )
        n_pick = min(3, len(sents)) if len(sents) > 3 else max(1, len(sents))
        if y_gold is not None and len(sents) > 0 and y_gold.sum() > 0:
            m_t = engine.sentence_selection_classification_metrics(tfidf_sc, y_gold, n_pick)
            m_h = engine.sentence_selection_classification_metrics(hybrid_sc, y_gold, n_pick)
            print("\n[Sentence-level metrics vs reference — TF-IDF top-k selection]")
            if m_t:
                print(
                    f"  Accuracy={m_t['accuracy']:.4f}  Precision={m_t['precision']:.4f}  "
                    f"Recall={m_t['recall']:.4f}  F1={m_t['f1']:.4f}"
                )
                print(f"  Confusion matrix [ [TN FP], [FN TP] ] labels order 0,1: {m_t['confusion_matrix']}")
            print("\n[Sentence-level metrics vs reference — Hybrid top-k selection]")
            if m_h:
                print(
                    f"  Accuracy={m_h['accuracy']:.4f}  Precision={m_h['precision']:.4f}  "
                    f"Recall={m_h['recall']:.4f}  F1={m_h['f1']:.4f}"
                )
                print(f"  Confusion matrix [ [TN FP], [FN TP] ]: {m_h['confusion_matrix']}")
        else:
            print(
                "\n[Sentence-level classification metrics skipped: no positive ROUGE labels "
                "or empty sentences — try another article or lower threshold in code.]"
            )

     
        score_tfidf = engine.evaluate(reference_text, tfidf_res)
        score_hybrid = engine.evaluate(reference_text, hybrid_res)
        score_bart = engine.evaluate(reference_text, bart_res)

        ratio_tfidf = engine.compression_ratio(article_text, tfidf_res)
        ratio_hybrid = engine.compression_ratio(article_text, hybrid_res)
        ratio_bart = engine.compression_ratio(article_text, bart_res)

        key_tfidf = engine.keyword_recall(article_text, tfidf_res)
        key_hybrid = engine.keyword_recall(article_text, hybrid_res)
        key_bart = engine.keyword_recall(article_text, bart_res)

        manual_tfidf = engine.manual_quality_score(article_text, tfidf_res, reference_text)
        manual_hybrid = engine.manual_quality_score(article_text, hybrid_res, reference_text)
        manual_bart = engine.manual_quality_score(article_text, bart_res, reference_text)

        aggregate["tfidf_rougeL"].append(score_tfidf['rougeL'].fmeasure)
        aggregate["hybrid_rougeL"].append(score_hybrid['rougeL'].fmeasure)
        aggregate["bart_rougeL"].append(score_bart['rougeL'].fmeasure)
        aggregate["tfidf_ratio"].append(ratio_tfidf)
        aggregate["hybrid_ratio"].append(ratio_hybrid)
        aggregate["bart_ratio"].append(ratio_bart)
        aggregate["tfidf_key"].append(key_tfidf)
        aggregate["hybrid_key"].append(key_hybrid)
        aggregate["bart_key"].append(key_bart)
        aggregate["tfidf_manual"].append(manual_tfidf)
        aggregate["hybrid_manual"].append(manual_hybrid)
        aggregate["bart_manual"].append(manual_bart)

        print("\nPerformance Metrics:")
        print(f"- TF-IDF Baseline: {score_tfidf['rougeL'].fmeasure:.4f}")
        print(f"- Hybrid TF-IDF+Embeddings: {score_hybrid['rougeL'].fmeasure:.4f}")
        print(f"- BART Advanced:   {score_bart['rougeL'].fmeasure:.4f}")
        print(f"- Compression Ratio (TF-IDF / Hybrid / BART): {ratio_tfidf:.3f} / {ratio_hybrid:.3f} / {ratio_bart:.3f}")
        print(f"- Keyword Recall (TF-IDF / Hybrid / BART): {key_tfidf:.3f} / {key_hybrid:.3f} / {key_bart:.3f}")
        print(f"- Manual Quality 1-5 (TF-IDF / Hybrid / BART): {manual_tfidf:.1f} / {manual_hybrid:.1f} / {manual_bart:.1f}")

    print("\n" + "="*60)
    print("Average Results Across Processed Articles")
    print("="*60)
    print(f"ROUGE-L: TF-IDF={np.mean(aggregate['tfidf_rougeL']):.4f}, Hybrid={np.mean(aggregate['hybrid_rougeL']):.4f}, BART={np.mean(aggregate['bart_rougeL']):.4f}")
    print(f"Compression Ratio: TF-IDF={np.mean(aggregate['tfidf_ratio']):.3f}, Hybrid={np.mean(aggregate['hybrid_ratio']):.3f}, BART={np.mean(aggregate['bart_ratio']):.3f}")
    print(f"Keyword Recall: TF-IDF={np.mean(aggregate['tfidf_key']):.3f}, Hybrid={np.mean(aggregate['hybrid_key']):.3f}, BART={np.mean(aggregate['bart_key']):.3f}")
    print(f"Manual Quality (1-5): TF-IDF={np.mean(aggregate['tfidf_manual']):.2f}, Hybrid={np.mean(aggregate['hybrid_manual']):.2f}, BART={np.mean(aggregate['bart_manual']):.2f}")

if __name__ == "__main__":
    run()
