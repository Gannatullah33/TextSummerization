"""
Streamlit GUI for the NLP summarization project.
Run from this folder: streamlit run app.py
"""

import streamlit as st
import pandas as pd

try:
    import altair as alt
except Exception:
    alt = None
from main import Summarizer


@st.cache_resource(show_spinner="Loading models (first time may download weights)…")
def load_engine():
    return Summarizer()


def main():
    st.set_page_config(page_title="Text Summarization — Project 4", layout="wide")

    st.markdown(
        """
        <style>
          .block-container { padding-top: 1.2rem; padding-bottom: 1.4rem; }
          div[data-testid="stMetricValue"] { font-size: 1.6rem; }
          div[data-testid="stMetricLabel"] { font-size: 0.95rem; }
          .soft-card {
            border: 1px solid rgba(120,120,120,0.25);
            border-radius: 14px;
            padding: 14px 14px;
            background: rgba(120,120,120,0.06);
          }
          .soft-muted { opacity: 0.75; }
          .pill {
            display: inline-block;
            padding: 4px 10px;
            border-radius: 999px;
            border: 1px solid rgba(120,120,120,0.25);
            background: rgba(120,120,120,0.06);
            font-size: 0.9rem;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )

    if "article" not in st.session_state:
        st.session_state.article = ""
    if "reference" not in st.session_state:
        st.session_state.reference = ""
    if "last" not in st.session_state:
        st.session_state.last = None

    top = st.container()
    with top:
        l, r = st.columns([0.72, 0.28], vertical_alignment="center")
        with l:
            st.title("Text summarization system")
            st.caption("Baseline: TF-IDF extractive · Improved: TF-IDF + embeddings · Advanced: BART")
        with r:
            st.markdown('<div class="soft-card">', unsafe_allow_html=True)
            st.markdown("**Tips**")
            st.markdown(
                '<div class="soft-muted">Try a longer article for best results. Add a reference summary to unlock evaluation metrics.</div>',
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

    with st.sidebar:
        st.header("Controls")
        n = st.slider("Extractive sentences (TF-IDF / hybrid)", 1, 10, 3)
        alpha = st.slider("Hybrid blend (TF-IDF weight)", 0.0, 1.0, 0.6, 0.05)
        show_all = st.toggle("Show all summaries", value=True)
        preferred = st.radio(
            "Preferred summary (for download/copy)",
            ["BART (advanced)", "Hybrid (TF-IDF + embeddings)", "TF-IDF (baseline)"],
            index=0,
        )
        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            run = st.button("Summarize", type="primary", use_container_width=True)
        with c2:
            if st.button("Clear", use_container_width=True):
                st.session_state.article = ""
                st.session_state.reference = ""
                st.session_state.last = None
                st.rerun()

        with st.expander("Example text", expanded=False):
            if st.button("Load demo article", use_container_width=True):
                st.session_state.article = (
                    "The exploration of Mars has been a focal point of space agencies for decades. "
                    "NASA's Perseverance rover, which landed in February 2021, is searching for signs of ancient microbial life "
                    "and collecting rock samples. Mars is often called the Red Planet because of iron oxide on its surface. "
                    "Scientists are interested in Mars because it once had liquid water, suggesting it could have supported life. "
                    "Future missions aim to bring these samples back to Earth for detailed analysis. "
                    "Elon Musk's SpaceX also has ambitious plans to send humans to Mars by the 2030s. "
                    "However, the harsh radiation and lack of a breathable atmosphere pose significant challenges for human colonization. "
                    "Despite these obstacles, the dream of becoming a multi-planetary species continues to drive innovation in aerospace engineering."
                )
                st.session_state.reference = (
                    "Mars exploration focuses on searching for ancient life, with NASA's Perseverance rover collecting samples. "
                    "While scientists study its watery past, future missions and SpaceX aim for human colonization. "
                    "However, radiation and atmospheric issues remain major challenges for missions to the Red Planet."
                )
                st.rerun()

    input_col, spacer, output_col = st.columns([0.46, 0.02, 0.52])
    with input_col:
        st.subheader("Input")
        st.session_state.article = st.text_area(
            "Article / long text",
            value=st.session_state.article,
            height=280,
            placeholder="Paste article text here…",
            label_visibility="visible",
        )
        st.session_state.reference = st.text_area(
            "Reference summary (optional)",
            value=st.session_state.reference,
            height=140,
            placeholder="If provided: ROUGE scores, keyword recall, and sentence-level Accuracy / Precision / Recall / F1 / confusion matrix.",
        )

    if run:
        article = str(st.session_state.article or "").strip()
        if not article:
            st.warning("Please enter some text to summarize.")
            return

        engine = load_engine()

        with st.spinner("Generating summaries…"):
            tfidf_res = engine.tfidf_summary(article, n=n)
            hybrid_res = engine.hybrid_extractive_summary(article, n=n, alpha=alpha)
            bart_res = engine.bart_summary(article)
            kw = engine.important_keywords(article, top_k=15)
            rank = engine.key_sentence_ranking(article, n=min(n, 5), alpha=alpha)

        st.session_state.last = {
            "article": article,
            "reference": str(st.session_state.reference or "").strip(),
            "n": n,
            "alpha": alpha,
            "tfidf": tfidf_res,
            "hybrid": hybrid_res,
            "bart": bart_res,
            "kw": kw,
            "rank": rank,
        }

    with output_col:
        st.subheader("Results")
        data = st.session_state.last
        if not data:
            st.info("Enter text on the left, then click **Summarize**.")
            return

        tfidf_res = data["tfidf"]
        hybrid_res = data["hybrid"]
        bart_res = data["bart"]
        article = data["article"]
        ref = data["reference"]
        rank = data["rank"]

        tabs = st.tabs(["Summaries", "Keywords & Sentences", "Evaluation"])

        def _pick(preference: str) -> str:
            if preference.startswith("BART"):
                return bart_res
            if preference.startswith("Hybrid"):
                return hybrid_res
            return tfidf_res

        chosen = _pick(preferred)

        with tabs[0]:
            wc_article = len(str(article).split())
            wc_pref = len(str(chosen).split())
            st.markdown(
                f'<span class="pill">Article words: <b>{wc_article}</b></span> &nbsp; '
                f'<span class="pill">Preferred summary words: <b>{wc_pref}</b></span>',
                unsafe_allow_html=True,
            )

            if show_all:
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.markdown("**TF-IDF (baseline)**")
                    st.text_area(" ", value=tfidf_res, height=240, label_visibility="collapsed")
                with c2:
                    st.markdown("**TF-IDF + embeddings (hybrid)**")
                    st.text_area("  ", value=hybrid_res, height=240, label_visibility="collapsed")
                with c3:
                    st.markdown("**BART (advanced)**")
                    st.text_area("   ", value=bart_res, height=240, label_visibility="collapsed")
            else:
                st.markdown(f"**{preferred}**")
                st.text_area("Summary", value=chosen, height=260, label_visibility="collapsed")

            b1, b2 = st.columns(2)
            with b1:
                st.download_button(
                    "Download preferred summary (.txt)",
                    data=chosen,
                    file_name="summary.txt",
                    mime="text/plain",
                    use_container_width=True,
                )
            with b2:
                st.download_button(
                    "Download all summaries (.txt)",
                    data=(
                        "TF-IDF (baseline)\n"
                        + tfidf_res
                        + "\n\nHybrid (TF-IDF + embeddings)\n"
                        + hybrid_res
                        + "\n\nBART (advanced)\n"
                        + bart_res
                        + "\n"
                    ),
                    file_name="summaries_all.txt",
                    mime="text/plain",
                    use_container_width=True,
                )

            st.divider()
            st.markdown("**Length comparison (words)**")
            len_df = pd.DataFrame(
                [
                    {"method": "Article", "words": wc_article},
                    {"method": "TF-IDF", "words": len(str(tfidf_res).split())},
                    {"method": "Hybrid", "words": len(str(hybrid_res).split())},
                    {"method": "BART", "words": len(str(bart_res).split())},
                ]
            )
            if alt is not None:
                chart = (
                    alt.Chart(len_df)
                    .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
                    .encode(
                        x=alt.X("method:N", sort=["Article", "TF-IDF", "Hybrid", "BART"]),
                        y=alt.Y("words:Q"),
                        color=alt.Color(
                            "method:N",
                            scale=alt.Scale(
                                domain=["Article", "TF-IDF", "Hybrid", "BART"],
                                range=["#64748B", "#60A5FA", "#34D399", "#A78BFA"],
                            ),
                            legend=None,
                        ),
                        tooltip=["method", "words"],
                    )
                    .properties(height=230)
                )
                st.altair_chart(chart, use_container_width=True)
            else:
                st.bar_chart(len_df.set_index("method")["words"])

        with tabs[1]:
            st.markdown("**Important keywords (document TF-IDF)**")
            kw = data.get("kw") or []
            if kw:
                st.write(", ".join(f"**{w}** ({s:.3f})" for w, s in kw))
            else:
                st.info("No keywords extracted (text too short after preprocessing).")

            st.divider()
            st.markdown("**Key sentences (scores)**")
            tc, hc = st.columns(2)
            with tc:
                st.markdown("*Highest TF-IDF importance*")
                for idx, sent, sc in rank.get("top_tfidf", []):
                    short = (sent[:220] + "…") if len(sent) > 220 else sent
                    with st.expander(f"Sentence #{idx} — score {sc:.4f}", expanded=False):
                        st.write(short)
            with hc:
                st.markdown("*Highest hybrid (TF-IDF + meaning)*")
                for idx, sent, sc in rank.get("top_hybrid", []):
                    short = (sent[:220] + "…") if len(sent) > 220 else sent
                    with st.expander(f"Sentence #{idx} — score {sc:.4f}", expanded=False):
                        st.write(short)

            st.divider()
            st.markdown("**Sentence score plot**")
            sents = rank.get("sentences", [])
            tfidf_sc = rank.get("tfidf_scores", [])
            hybrid_sc = rank.get("hybrid_scores", [])
            if len(sents) and len(tfidf_sc) == len(sents) and len(hybrid_sc) == len(sents):
                plot_df = pd.DataFrame(
                    {
                        "sentence_index": list(range(1, len(sents) + 1)),
                        "TF-IDF": tfidf_sc,
                        "Hybrid": hybrid_sc,
                    }
                ).melt("sentence_index", var_name="method", value_name="score")
                if alt is not None:
                    chart = (
                        alt.Chart(plot_df)
                        .mark_line(point=True)
                        .encode(
                            x=alt.X("sentence_index:Q", title="Sentence #"),
                            y=alt.Y("score:Q", title="Importance score"),
                            color=alt.Color(
                                "method:N",
                                scale=alt.Scale(domain=["TF-IDF", "Hybrid"], range=["#60A5FA", "#34D399"]),
                            ),
                            tooltip=["method", "sentence_index", alt.Tooltip("score:Q", format=".4f")],
                        )
                        .properties(height=260)
                    )
                    st.altair_chart(chart, use_container_width=True)
                else:
                    wide = pd.DataFrame({"TF-IDF": tfidf_sc, "Hybrid": hybrid_sc})
                    st.line_chart(wide)
            else:
                st.info("Not enough sentence score data to plot.")

        with tabs[2]:
            if not ref:
                st.info("Add a **reference summary** on the left to see ROUGE, keyword recall, and classification metrics.")
                return

            st.markdown("**Overlap metrics vs reference**")
            engine = load_engine()
            r_t = engine.evaluate(ref, tfidf_res)
            r_h = engine.evaluate(ref, hybrid_res)
            r_b = engine.evaluate(ref, bart_res)

            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("ROUGE-L (TF-IDF)", f"{r_t['rougeL'].fmeasure:.4f}")
                st.caption(f"ROUGE-1 F1: {r_t['rouge1'].fmeasure:.4f}")
            with m2:
                st.metric("ROUGE-L (hybrid)", f"{r_h['rougeL'].fmeasure:.4f}")
                st.caption(f"ROUGE-1 F1: {r_h['rouge1'].fmeasure:.4f}")
            with m3:
                st.metric("ROUGE-L (BART)", f"{r_b['rougeL'].fmeasure:.4f}")
                st.caption(f"ROUGE-1 F1: {r_b['rouge1'].fmeasure:.4f}")

            st.markdown("**ROUGE comparison plot**")
            rouge_df = pd.DataFrame(
                [
                    {"method": "TF-IDF", "metric": "ROUGE-1 F1", "score": float(r_t["rouge1"].fmeasure)},
                    {"method": "TF-IDF", "metric": "ROUGE-L F1", "score": float(r_t["rougeL"].fmeasure)},
                    {"method": "Hybrid", "metric": "ROUGE-1 F1", "score": float(r_h["rouge1"].fmeasure)},
                    {"method": "Hybrid", "metric": "ROUGE-L F1", "score": float(r_h["rougeL"].fmeasure)},
                    {"method": "BART", "metric": "ROUGE-1 F1", "score": float(r_b["rouge1"].fmeasure)},
                    {"method": "BART", "metric": "ROUGE-L F1", "score": float(r_b["rougeL"].fmeasure)},
                ]
            )
            if alt is not None:
                chart = (
                    alt.Chart(rouge_df)
                    .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
                    .encode(
                        x=alt.X("method:N", sort=["TF-IDF", "Hybrid", "BART"]),
                        y=alt.Y("score:Q", scale=alt.Scale(domain=[0, 1])),
                        color=alt.Color(
                            "metric:N",
                            scale=alt.Scale(domain=["ROUGE-1 F1", "ROUGE-L F1"], range=["#F59E0B", "#22C55E"]),
                        ),
                        column=alt.Column("metric:N", title=None),
                        tooltip=["method", "metric", alt.Tooltip("score:Q", format=".4f")],
                    )
                    .properties(height=240)
                )
                st.altair_chart(chart, use_container_width=True)
            else:
                pivot = rouge_df.pivot(index="method", columns="metric", values="score")
                st.bar_chart(pivot)

            a, b = st.columns(2)
            with a:
                st.markdown("**Compression ratio** (summary words / article words)")
                st.json(
                    {
                        "TF-IDF": f"{engine.compression_ratio(article, tfidf_res):.3f}",
                        "Hybrid": f"{engine.compression_ratio(article, hybrid_res):.3f}",
                        "BART": f"{engine.compression_ratio(article, bart_res):.3f}",
                    }
                )
            with b:
                st.markdown("**Keyword recall** (overlap of top TF-IDF terms)")
                st.json(
                    {
                        "TF-IDF": f"{engine.keyword_recall(article, tfidf_res):.3f}",
                        "Hybrid": f"{engine.keyword_recall(article, hybrid_res):.3f}",
                        "BART": f"{engine.keyword_recall(article, bart_res):.3f}",
                    }
                )

            sents = rank["sentences"]
            tfidf_sc = rank["tfidf_scores"]
            hybrid_sc = rank["hybrid_scores"]
            y_gold = engine.gold_sentence_labels_from_reference(sents, ref)
            if y_gold is not None and int(y_gold.sum()) == 0:
                y_gold = engine.gold_sentence_labels_from_reference(
                    sents, ref, rouge1_f1_threshold=0.06
                )
            n_pick = min(data["n"], len(sents)) if len(sents) > data["n"] else max(1, len(sents))

            st.divider()
            st.markdown("**Sentence-level classification (extractive selection vs reference)**")
            st.caption(
                "Each sentence is labeled important if ROUGE-1 vs reference exceeds a threshold; "
                "the model predicts important = among top-k scores."
            )
            if y_gold is not None and len(sents) > 0 and int(y_gold.sum()) > 0:
                mt = engine.sentence_selection_classification_metrics(tfidf_sc, y_gold, n_pick)
                mh = engine.sentence_selection_classification_metrics(hybrid_sc, y_gold, n_pick)
                if mt and mh:
                    d1, d2 = st.columns(2)
                    with d1:
                        st.markdown("**TF-IDF top-k**")
                        st.json(
                            {
                                "accuracy": mt["accuracy"],
                                "precision": mt["precision"],
                                "recall": mt["recall"],
                                "f1": mt["f1"],
                                "confusion_matrix_labels_01": mt["confusion_matrix"],
                            }
                        )
                    with d2:
                        st.markdown("**Hybrid top-k**")
                        st.json(
                            {
                                "accuracy": mh["accuracy"],
                                "precision": mh["precision"],
                                "recall": mh["recall"],
                                "f1": mh["f1"],
                                "confusion_matrix_labels_01": mh["confusion_matrix"],
                            }
                        )
            else:
                st.info("Could not build sentence labels from this reference (try a longer reference summary).")


if __name__ == "__main__":
    main()
