import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from sklearn.metrics.pairwise import cosine_similarity
import openai
import os
import re

# ---------------------------
# App Header
# ---------------------------
st.title("Jira ↔ TMS Semantic Mapper")
st.caption("Developed by Pamidi Bala Sumanth (AI-powered)")

# ---------------------------
# OpenAI API Key
# ---------------------------
api_key = st.text_input("Enter your OpenAI API key", type="password")
if api_key:
    openai.api_key = api_key
elif os.getenv("OPENAI_API_KEY"):
    openai.api_key = os.getenv("OPENAI_API_KEY")

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
# Keyword-based platform and component mapping
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
# Get embeddings
# ---------------------------
def get_embedding(text, model="text-embedding-3-small"):
    if not text or not isinstance(text, str) or not text.strip():
        return None
    result = openai.Embedding.create(
        input=text,
        model=model
    )
    return np.array(result['data'][0]['embedding'])

# ---------------------------
# Main
# ---------------------------
if jira_file and tms_file and openai.api_key:
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

    # Compute embeddings with progress bar
    progress = st.progress(0, text="🔄 Generating embeddings...")

    jira_embs, tms_embs = [], []
    for i, txt in enumerate(jira["__bug_text__"]):
        jira_embs.append(get_embedding(txt))
        progress.progress(int((i+1)/len(jira)*50), text="🔄 Embedding Jira bugs...")

    for i, txt in enumerate(tms["__full_text__"]):
        tms_embs.append(get_embedding(txt))
        progress.progress(50 + int((i+1)/len(tms)*50), text="🔄 Embedding TMS cases...")

    progress.empty()

    jira_embs = np.vstack([e for e in jira_embs if e is not None])
    tms_embs = np.vstack([e for e in tms_embs if e is not None])

    # Compute similarity matrix
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
