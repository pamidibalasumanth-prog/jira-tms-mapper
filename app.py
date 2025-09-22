import os
import re
import time
import random
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

def get_embeddings_batched(texts, client, model="text-embedding-3-small", batch_size=100):
    """Get embeddings in safe batches with retry and backoff"""
    vectors = []
    for i in range(0, len(texts), batch_size):
        batch = [t for t in texts[i:i+batch_size] if isinstance(t, str) and t.strip()]
        if not batch:
            continue

        success = False
        while not success:
            try:
                resp = client.embeddings.create(model=model, input=batch)
                vectors.extend([d.embedding for d in resp.data])
                success = True
            except Exception:
                time.sleep(random.uniform(5, 15))  # retry wait
    return vectors

# ------------------------
# Streamlit UI
# ------------------------

st.title("Jira ↔ TMS Mapper (with New Test Case Creation)")
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
    with st.spinner("🔎 Generating embeddings for Jira and TMS cases in batches..."):
        jira_embs = get_embeddings_batched(jira["__bug_text__"].tolist(), client)
        tms_embs = get_embeddings_batched(tms["__case_text__"].tolist(), client)

    jira_matrix = np.array(jira_embs)
    tms_matrix = np.array(tms_embs)

    sims = cosine_similarity(jira_matrix, tms_matrix)

    # ------------------------
    # Mapping
    # ------------------------
    best_matches = []
    new_cases = []

    for i, row in jira.iterrows():
        sim_scores = sims[i]
        top_idx = np.argsort(sim_scores)[::-1]

        best_id = None
        best_score = 0

        for idx in top_idx[:1]:  # only best
            score = sim_scores[idx]
            if score >= threshold:
                best_id = tms[id_col].iloc[idx]
                best_score = score

        if best_id:
            best_matches.append(best_id)
        else:
            # No match found → mark as needing a new test case
            best_matches.append("Test case needs to be added – no match found")

            new_cases.append({
                "Title": row["Summary"],
                "Description": row.get("Description", row["Summary"]),
                "Folder": row.get("Component", "Unassigned"),
                "Priority": row.get("Priority", ""),
                "Suggested Steps": "Reproduce steps based on bug report",
                "Source Bug": row["Issue key"]
            })

    jira["TMS ID"] = best_matches

    # ------------------------
    # Export
    # ------------------------
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        jira.to_excel(writer, index=False, sheet_name="Mapped")

        if new_cases:
            pd.DataFrame(new_cases).to_excel(writer, index=False, sheet_name="New TMS Cases")

    processed_data = output.getvalue()

    st.success(f"✅ Mapping complete! {jira['TMS ID'].str.contains('Test case needs').sum()} new test cases suggested.")
    st.dataframe(jira[["Issue key", "Summary", "TMS ID"]].head(20))

    st.download_button(
        label="Download Result",
        data=processed_data,
        file_name="Jira_C_List_TMS_Final.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
