import os
import re
import time
import random
import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity

# =========================
# UI + Page Setup
# =========================
st.set_page_config(page_title="Jira ↔ TMS Mapper", page_icon="📝", layout="wide")
st.title("Jira ↔ TMS Mapper")
st.caption("Developed by Pamidi Bala Sumanth • AI-powered matching + auto new-case drafting")

# ---- Inputs
api_key = st.text_input("OpenAI API key", type="password", help="Use org key for best reliability.")
jira_file = st.file_uploader("Upload Jira C List (CSV/XLSX)", type=["csv", "xlsx"])
tms_file  = st.file_uploader("Upload TMS Export (Excel)", type=["xlsx"])
threshold = st.slider("Similarity threshold (cosine)", 0.0, 1.0, 0.55, 0.01,
                      help="Lower = more mappings (riskier), Higher = fewer mappings (stricter)")

# =========================
# Hardcoded Rules
# =========================
# Platform: single (first match wins)
PLATFORM_RULES = [
    ("salesforce", "Salesforce"),
    ("sap",        "SAP"),
    ("mobile",     "Mobile"),
    ("android",    "Mobile"),
    ("ios",        "Mobile"),
    ("desktop",    "Desktop"),
    ("api",        "API"),
    ("endpoint",   "API"),
    ("rest",       "API"),
    ("web",        "Web"),
    ("browser",    "Web"),
]

# Component: multiple, comma-separated
COMPONENT_RULES = {
    "add-on": "Add-Ons",
    "addons": "Add-Ons",
    "windows .net": "Windows - .NET",
    "windows java": "Windows - Java",
    "user": "User Management",
    "auth": "User Management",
    "login": "User Management",
    "tunnel": "Tunnel",
    "test plan": "Test Plans and Runs",
    "plan run": "Test Plans and Runs",
    "authoring": "Test Authoring",
    "test author": "Test Authoring",
    "terminal": "Terminal & Agent",
    "agent": "Terminal & Agent",
    "sap": "SAP",
    "salesforce": "SalesForce",
    "recorder": "Recorder (Browser)",
    "record": "Recorder (Browser)",
    "api": "Public APIs",
    "project": "Projects",
    "nlp": "NLP",
    "editor": "Live Editor",
    "live editor": "Live Editor",
    "integration": "Integrations",
    "execute": "Execution",
    "execution": "Execution",
    "doc": "Documentation",
    "documentation": "Documentation",
    "co-pilot": "Co-pilot",
    "copilot": "Co-pilot",
    "billing": "Billing & Licenses",
    "license": "Billing & Licenses",
    "autonomous": "Autonomous",
    "heal": "Auto Heal",
    "atto live": "Atto Live Editor",
    "atto generator": "Atto Generator",
    "atto analyzer": "Atto Analyzer",
    "ai ": "AI Research",
    " ai": "AI Research",
}

# When creating a new test case, pick a folder from components (fallback to 'Unassigned')
def choose_folder_from_components(components: str) -> str:
    return components.split(",")[0].strip() if components else "Unassigned"

# =========================
# Helpers
# =========================
def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    # remove bracket tags like [Prod], [Request], etc.
    text = re.sub(r"\[.*?\]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()

def platform_from_text(text: str) -> str:
    t = text.lower()
    for kw, plat in PLATFORM_RULES:
        if kw in t:
            return plat
    return ""  # leave blank if unknown

def components_from_text(text: str) -> str:
    t = text.lower()
    comps = []
    for kw, comp in COMPONENT_RULES.items():
        if kw in t:
            comps.append(comp)
    # unique + stable sort
    comps = sorted(set(comps))
    return ", ".join(comps)

def batched_embeddings(client: OpenAI, texts, model="text-embedding-3-small", batch_size=100):
    """
    Safely embed a large list: batching + simple retry/backoff.
    Returns list[np.array] aligned to non-empty inputs; empty/whitespace strings produce a zero vector.
    """
    vectors = []
    for i in range(0, len(texts), batch_size):
        batch_raw = texts[i:i+batch_size]
        # For empty strings, use a zero vector placeholder to keep indices aligned
        nonempty_idx = []
        nonempty_texts = []
        for idx, t in enumerate(batch_raw):
            if isinstance(t, str) and t.strip():
                nonempty_idx.append(idx)
                nonempty_texts.append(t)
        # Call API for non-empty; for empty push zeros later
        success = False
        while not success:
            try:
                if nonempty_texts:
                    resp = client.embeddings.create(model=model, input=nonempty_texts)
                    emb_list = [np.array(d.embedding, dtype=np.float32) for d in resp.data]
                else:
                    emb_list = []
                # reconstruct batch with zero-vectors for empties
                k = 0
                for j in range(len(batch_raw)):
                    if j in nonempty_idx:
                        vectors.append(emb_list[k])
                        k += 1
                    else:
                        # zero vector length must match model dims; 1536 for text-embedding-3-small
                        vectors.append(np.zeros(1536, dtype=np.float32))
                success = True
            except Exception:
                time.sleep(random.uniform(5, 15))
    return vectors

def ensure_columns(df: pd.DataFrame, cols):
    for c in cols:
        if c not in df.columns:
            df[c] = ""
    return df

# =========================
# Main
# =========================
if api_key and jira_file and tms_file:
    client = OpenAI(api_key=api_key)

    # ---- Load Jira
    jira = pd.read_csv(jira_file) if jira_file.name.endswith(".csv") else pd.read_excel(jira_file)
    jira = ensure_columns(jira, ["Summary", "Description", "Issue Type", "Issue key", "Assignee",
                                 "Reporter", "Priority", "Status", "Review Assignee",
                                 "Review Comments", "TMS Folder Name", "Priority Changed"])

    # ---- Load TMS
    try:
        tms = pd.read_excel(tms_file, sheet_name="Test Cases")
    except Exception:
        tms = pd.read_excel(tms_file, sheet_name=0)

    # Flexible detection of TMS ID/Title columns
    try:
        tms_id_col = [c for c in tms.columns if "id" in c.lower()][0]
    except IndexError:
        st.error("Could not find TMS ID column. Please ensure your TMS sheet has an 'ID' column.")
        st.stop()
    try:
        tms_title_col = [c for c in tms.columns if "title" in c.lower()][0]
    except IndexError:
        st.error("Could not find TMS Title column. Please ensure your TMS sheet has a 'Title' column.")
        st.stop()

    # ---- Build matching texts
    jira["__bug_text__"] = (jira["Summary"].fillna("") + " " + jira["Description"].fillna("")).apply(clean_text)
    tms["__case_text__"] = (
        tms[tms_title_col].fillna("") + " " +
        tms.get("Description", "").fillna("") + " " +
        tms.get("Steps", "").fillna("") + " " +
        tms.get("Expected Results", "").fillna("")
    ).apply(clean_text)

    # ---- Embeddings (batched + retried)
    with st.spinner("🔎 Generating embeddings (batched & retried)…"):
        jira_vecs = batched_embeddings(client, jira["__bug_text__"].tolist(), model="text-embedding-3-small", batch_size=100)
        tms_vecs  = batched_embeddings(client, tms["__case_text__"].tolist(), model="text-embedding-3-small", batch_size=100)

    jira_mat = np.vstack(jira_vecs)
    tms_mat  = np.vstack(tms_vecs)

    # ---- Similarity matrix
    sims = cosine_similarity(jira_mat, tms_mat)

    # ---- For each bug: pick best match OR mark as new test case required
    tms_ids = []
    sim_scores = []
    platforms = []
    components = []
    new_cases_rows = []

    for i, row in jira.iterrows():
        # Platform & Components (from bug text)
        p_text = jira.at[i, "__bug_text__"]
        plat = platform_from_text(p_text)
        comp = components_from_text(p_text)
        platforms.append(plat)
        components.append(comp)

        # Best semantic match
        row_scores = sims[i]
        best_idx = int(np.argmax(row_scores))
        best_score = float(row_scores[best_idx]) if len(row_scores) else 0.0

        # Decide mapping
        if best_score >= threshold:
            tms_ids.append(str(tms.at[best_idx, tms_id_col]))
            sim_scores.append(best_score)
            # If we have folder in TMS, reuse it
            if "Folder" in tms.columns and not pd.isna(tms.at[best_idx, "Folder"]):
                jira.at[i, "TMS Folder Name"] = tms.at[best_idx, "Folder"]
        else:
            tms_ids.append("Test case needs to be added – no match found")
            sim_scores.append(best_score)

            # Prepare draft case for second sheet
            new_cases_rows.append({
                "Title": row["Summary"],
                "Description": row["Description"] if pd.notna(row["Description"]) and str(row["Description"]).strip() else row["Summary"],
                "Folder": choose_folder_from_components(comp),
                "Platform": plat,
                "Priority": row["Priority"],
                "Suggested Steps": "Reproduce steps based on bug report.",
                "Source Bug": row["Issue key"]
            })

    # ---- Fill columns into Jira DF
    jira["TMS ID"] = tms_ids
    jira["Similarity Score"] = sim_scores
    jira["Platform"] = platforms
    jira["Component"] = components

    # ---- Final column order for export
    desired_order = [
        "Issue Type", "Issue key", "Summary", "Assignee", "Reporter",
        "Priority", "Status", "Review Assignee", "Review Comments",
        "TMS ID", "Component", "Platform", "TMS Folder Name",
        "Priority Changed", "Similarity Score"
    ]
    output_df = jira[[c for c in desired_order if c in jira.columns]].copy()

    # ---- Preview
    mapped_count = int((output_df["TMS ID"] != "Test case needs to be added – no match found").sum()) if "TMS ID" in output_df.columns else 0
    st.success(f"✅ Complete: {mapped_count}/{len(output_df)} mapped to existing TMS. "
               f"{len(output_df) - mapped_count} need new test cases.")
    st.dataframe(output_df.head(30), use_container_width=True)

    # ---- Export (two sheets)
    out = BytesIO()
    with pd.ExcelWriter(out, engine="xlsxwriter") as writer:
        output_df.to_excel(writer, index=False, sheet_name="Mapped")
        if new_cases_rows:
            pd.DataFrame(new_cases_rows, columns=[
                "Title", "Description", "Folder", "Platform", "Priority", "Suggested Steps", "Source Bug"
            ]).to_excel(writer, index=False, sheet_name="New TMS Cases")

    st.download_button(
        "📥 Download Excel",
        out.getvalue(),
        file_name="Jira_C_List_TMS_Final.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
