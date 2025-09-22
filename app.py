import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
import os
import hashlib
import pickle

# ---------------------------
# App Header
# ---------------------------
st.title("Jira ↔ TMS Semantic Mapper (Optimized)")
st.caption("Developed by Pamidi Bala Sumanth – AI-powered with caching")

# ---------------------------
# OpenAI API Key
# ---------------------------
api_key = st.text_input("Enter your OpenAI API key", type="password")
if api_key:
    client = OpenAI(api_key=api_key)
elif os.getenv("OPENAI_API_KEY"):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
else:
    client = None

# ---------------------------
# File Uploads
# ---------------------------
jira_file = st.file_uploader("Upload Jira C List (CSV/XLSX)", type=["csv", "xlsx"])
tms_file  = st.file_uploader("Upload TMS Export (Excel)", type=["xlsx"])

# ---------------------------
# Similarity Threshold
# ---------------------------
threshold = st.slider("Select similarity threshold (cosine)", 0.0, 1.0, 0.70, 0.01)

# ---------------------------
# Platform & Component Mapping
# ---------------------------
platform_keywords = {
    "mobile": "Mobile",
    "desktop": "Desktop",
    "api": "API",
    "salesforce": "Salesforce",
    "sap": "SAP",
    "web": "Web"
}

component_keywords = {
    "add-on": "Add-Ons",
    "windows .net": "Windows - .NET",
    "windows java": "Windows - Java",
    "user": "User Management",
    "tunnel": "Tunnel",
    "test plan": "Test Plans and Runs",
    "authoring": "Test Authoring",
    "terminal": "Terminal & Agent",
    "sap": "SAP",
    "salesforce": "SalesForce",
    "recorder": "Recorder (Browser)",
    "api": "Public APIs",
    "project": "Projects",
    "nlp": "NLP",
    "editor": "Live Editor, Atto Live Editor",
    "integration": "Integrations",
    "execution": "Execution",
    "doc": "Documentation",
    "co-pilot": "Co-pilot",
    "billing": "Billing & Licenses",
    "autonomous": "Autonomous",
    "heal": "Auto Heal",
    "atto": "Atto Live Editor, Atto Generator, Atto Analyzer",
    "ai": "AI Research"
}

def platform_from_text(text):
    for kw, plat in platform_keywords.items():
        if kw in text.lower():
            return plat
    return ""

def components_from_text(text):
    comps = []
    for kw, comp in component_keywords.items():
        if kw in text.lower():
            comps.extend([c.strip() for c in comp.split(",")])
    return ", ".join(sorted(set(comps)))

# ---------------------------
# Caching helper
# ---------------------------
def cache_key(texts, model):
    concat = "|".join(texts)
    return hashlib.md5((concat + model).encode("utf-8")).hexdigest()

def load_cached_embeddings(key):
    fname = f"cache_{key}.pkl"
    if os.path.exists(fname):
        with open(fname, "rb") as f:
            return pickle.load(f)
    return None

def save_cached_embeddings(key, embs):
    fname = f"cache_{key}.pkl"
    with open(fname, "wb") as f:
        pickle.dump(embs, f)

# ---------------------------
# Batch embedding
# ---------------------------
def get_embeddings(texts, model="text-embedding-3-small"):
    key = cache_key(texts, model)
    cached = load_cached_embeddings(key)
    if cached is not None:
        return cached

    emb = client.embeddings.create(model=model, input=texts)
    vectors = [np.array(r.embedding) for r in emb.data]

    save_cached_embeddings(key, vectors)
    return vectors

# ---------------------------
# Main
# ---------------------------
if jira_file and tms_file and client:
    # Load Jira
    jira = pd.read_csv(jira_file) if jira_file.name.endswith(".csv") else pd.read_excel(jira_file)
    jira["__bug_text__"] = (jira.get("Summary","").fillna("").astype(str) + " " +
                            jira.get("Description","").fillna("").astype(str))

    # Load TMS
    try:
        tms = pd.read_excel(tms_file, sheet_name="Test Cases")
    except Exception:
        tms = pd.read_excel(tms_file)

    tms["__full_text__"] = (
        tms.get("Title","").fillna("").astype(str) + " " +
        tms.get("Description","").fillna("").astype(str) + " " +
        tms.get("Steps","").fillna("").astype(str) + " " +
        tms.get("Expected Results","").fillna("").astype(str)
    )

    # Embedding in 2 calls
    with st.spinner("🔄 Generating embeddings (cached if possible)..."):
        jira_embs = get_embeddings(jira["__bug_text__"].tolist())
        tms_embs  = get_embeddings(tms["__full_text__"].tolist())

    jira_embs = np.vstack(jira_embs)
    tms_embs = np.vstack(tms_embs)

    # Similarity
    sim_matrix = cosine_similarity(jira_embs, tms_embs)

    # One-to-one greedy assignment
    assigned_bug, assigned_tms, chosen = set(), set(), {}
    edges = []
    for i in range(sim_matrix.shape[0]):
        for j in range(sim_matrix.shape[1]):
            score = sim_matrix[i, j]
            if score >= threshold:
                edges.append((score, i, j))
    edges.sort(reverse=True)

    for score, b_idx, t_idx in edges:
        if b_idx in assigned_bug or t_idx in assigned_tms:
            continue
        assigned_bug.add(b_idx)
        assigned_tms.add(t_idx)
        chosen[b_idx] = (t_idx, score)

    # Fill output
    jira["TMS ID"] = ""
    jira["Similarity Score"] = 0.0
    jira["Platform"] = ""
    jira["Component"] = ""
    jira["TMS Folder Name"] = ""
    jira["Priority Changed"] = ""

    for b_idx in range(len(jira)):
        text = jira.at[b_idx, "__bug_text__"]
        jira.at[b_idx, "Platform"] = platform_from_text(text)
        jira.at[b_idx, "Component"] = components_from_text(text)

        if b_idx in chosen:
            t_idx, score = chosen[b_idx]
            jira.at[b_idx, "TMS ID"] = tms.at[t_idx, "ID"]
            jira.at[b_idx, "Similarity Score"] = score
            if "Folder" in tms.columns:
                jira.at[b_idx, "TMS Folder Name"] = tms.at[t_idx, "Folder"]

    # Export
    desired_order = [
        "Issue Type", "Issue key", "Summary", "Assignee", "Reporter",
        "Priority", "Status", "Review Assignee", "Review Comments",
        "TMS ID", "Component", "Platform", "TMS Folder Name",
        "Priority Changed", "Similarity Score"
    ]
    output_df = jira[[c for c in desired_order if c in jira.columns]].copy()

    st.success("✅ Semantic mapping complete!")
    st.info(f"Mapped {output_df['TMS ID'].astype(bool).sum()} out of {len(output_df)} bugs")

    st.subheader("Preview of Mapped Data")
    st.dataframe(output_df.head(50), use_container_width=True)

    outbuf = BytesIO()
    with pd.ExcelWriter(outbuf, engine="xlsxwriter") as writer:
        output_df.to_excel(writer, index=False, sheet_name="Mapped")
    st.download_button(
        "📥 Download Result",
        outbuf.getvalue(),
        file_name="Jira_C_List_TMS_Semantic.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
