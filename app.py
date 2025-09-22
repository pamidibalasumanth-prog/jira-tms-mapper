import os
import re
import streamlit as st
import pandas as pd
from io import BytesIO
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ------------------------
# Helpers
# ------------------------

def clean_text(text: str) -> str:
    """Normalize Jira/TMS text for better matching"""
    if not isinstance(text, str):
        return ""
    # remove [Prod], [Request], etc.
    text = re.sub(r"\[.*?\]", "", text)
    return text.lower().strip()

def get_embeddings(texts, client, model="text-embedding-3-small"):
    """Batch embeddings with OpenAI"""
    if not texts:
        return []
    response = client.embeddings.create(
        model=model,
        input=texts
    )
    return [d.embedding for d in response.data]

# ------------------------
# Streamlit UI
# ------------------------

st.title("Jira ↔ TMS Mapper")
st.markdown("**Developed by Pamidi Bala Sumanth**")

api_key = st.text_input("Enter your OpenAI API key", type="password")

jira_file = st.file_uploader("Upload Jira C List (CSV/XLSX)", type=["csv", "xlsx"])
tms_file = st.file_uploader("Upload TMS Export (Excel)", type=["xlsx"])
threshold = st.slider("Select similarity threshold (cosine)", 0.0, 1.0, 0.5, 0.05)

if api_key and jira_file and tms_file:
    client = OpenAI(api_key=api_key)

    # ------------------------
    # Load Data
    # ------------------------
    if jira_file.name.endswith(".csv"):
        jira = pd.read_csv(jira_file)
    else:
        jira = pd.read_excel(jira_file)

    tms = pd.read_excel(tms_file, sheet_name=0)

    # Flexible detection of TMS columns
    id_col = [c for c in tms.columns if "id" in c.lower()][0]
    title_col = [c for c in tms.columns if "title" in c.lower()][0]

    # Clean fields
    jira["__bug_text__"] = jira["Summary"].apply(clean_text)
    tms["__case_text__"] = tms[title_col].apply(clean_text)

    # ------------------------
    # Embeddings
    # ------------------------
    with st.spinner("🔎 Generating embeddings for Jira and TMS cases..."):
        jira_embs = get_embeddings(jira["__bug_text__"].tolist(), client)
        tms_embs = get_embeddings(tms["__case_text__"].tolist(), client)

    jira_matrix = np.array(jira_embs)
    tms_matrix = np.array(tms_embs)

    sims = cosine_similarity(jira_matrix, tms_matrix)

    # ------------------------
    # Mapping
    # ------------------------
    best_matches = []
    alt_matches = []

    for i, row in jira.iterrows():
        sim_scores = sims[i]
        top_idx = np.argsort(sim_scores)[::-1]  # descending

        best_id = None
        best_score = 0
        candidates = []

        for idx in top_idx[:3]:  # keep top 3
            score = sim_scores[idx]
            if score >= threshold:
                candidates.append(f"{tms[id_col].iloc[idx]} ({score:.2f})")
                if best_id is None:  # first valid
                    best_id = tms[id_col].iloc[idx]
                    best_score = score

        best_matches.append(best_id if best_id else "")
        alt_matches.append(", ".join(candidates))

    jira["TMS ID"] = best_matches
    jira["Alt Matches"] = alt_matches

    # ------------------------
    # Export
    # ------------------------
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        jira.to_excel(writer, index=False, sheet_name="Mapped")
    processed_data = output.getvalue()

    st.success(f"✅ Mapping complete! {jira['TMS ID'].astype(bool).sum()} of {len(jira)} bugs mapped.")
    st.dataframe(jira[["Issue key", "Summary", "TMS ID", "Alt Matches"]].head(20))

    st.download_button(
        label="Download Result",
        data=processed_data,
        file_name="Jira_C_List_TMS_Final.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
