import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from rapidfuzz import process, fuzz
import re

# ---------------------------
# App Header
# ---------------------------
st.title("Jira ↔ TMS Mapper")
st.caption("Developed by Pamidi Bala Sumanth")

# ---------------------------
# File Uploads
# ---------------------------
jira_file = st.file_uploader("Upload Jira C List (CSV/XLSX)", type=["csv", "xlsx"])
tms_file  = st.file_uploader("Upload TMS Export (Excel)",       type=["xlsx"])

# ---------------------------
# Mapping Controls
# ---------------------------
threshold = st.slider("Minimum similarity to accept a mapping", 0, 100, 60, step=5)
use_domain_filter = st.checkbox("Use domain-aware filtering (recommended)", value=True)
topk = st.slider("Top candidates per bug (higher = slower, maybe better)", 1, 15, 7)

# ---------------------------
# Keyword maps (Platform = single; Components = multi)
# ---------------------------
platform_keywords = [
    ("mobile",      "Mobile"),
    ("desktop",     "Desktop"),
    ("api",         "API"),
    ("salesforce",  "Salesforce"),
    ("sap",         "SAP"),
    ("web",         "Web"),
]

component_map = {
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
    "ai": "AI Research",
}

# Reusable utilities
def text_or_empty(df, col):
    return df[col] if col in df.columns else ""

def components_from_text(text: str) -> str:
    t = text.lower()
    comps = []
    for kw, comps_str in component_map.items():
        if kw in t:
            comps.extend([c.strip() for c in comps_str.split(",")])
    # unique & sorted for stable output
    return ", ".join(sorted(set(comps)))

def platform_from_text(text: str) -> str:
    t = text.lower()
    for kw, plat in platform_keywords:
        if kw in t:
            return plat
    return ""  # leave blank if unknown

def extract_domain_tags(text: str) -> set:
    """Lightweight domain tags to help filter TMS candidates"""
    t = text.lower()
    keys = [
        "salesforce","sap","recorder","api","terminal","agent","integration",
        "billing","nlp","tunnel","execution","editor","project","author","test plan",
        "documentation","co-pilot","autonomous","heal","atto","ai","windows"
    ]
    return {k for k in keys if k in t}

# ---------------------------
# Main
# ---------------------------
if jira_file and tms_file:
    # Load Jira
    jira = pd.read_csv(jira_file) if jira_file.name.endswith(".csv") else pd.read_excel(jira_file)

    # Load TMS (use "Test Cases" if present, else first sheet)
    try:
        tms = pd.read_excel(tms_file, sheet_name="Test Cases")
    except Exception:
        tms = pd.read_excel(tms_file)

    # Columns we rely on
    ID_COL    = "ID"      # TMS ID
    TITLE_COL = "Title"   # TMS Title

    # Build rich TMS text
    tms = tms.copy()
    tms["__full_text__"] = (
        tms[TITLE_COL].fillna("").astype(str) + " " +
        text_or_empty(tms, "Description").fillna("").astype(str) + " " +
        text_or_empty(tms, "Steps").fillna("").astype(str) + " " +
        text_or_empty(tms, "Expected Results").fillna("").astype(str) + " " +
        text_or_empty(tms, "Folder").fillna("").astype(str) + " " +
        text_or_empty(tms, "Labels").fillna("").astype(str)
    )

    # Prepare Jira working text
    jira = jira.copy()
    jira["__bug_text__"] = (jira.get("Summary","").fillna("").astype(str) + " " +
                            jira.get("Description","").fillna("").astype(str))

    # Prepare output columns
    if "Component" not in jira.columns: jira["Component"] = ""
    if "Platform"  not in jira.columns: jira["Platform"]  = ""
    if "TMS ID"    not in jira.columns: jira["TMS ID"]    = ""
    if "Similarity Score" not in jira.columns: jira["Similarity Score"] = 0.0
    if "TMS Folder Name"  not in jira.columns: jira["TMS Folder Name"] = ""
    if "Priority Changed" not in jira.columns: jira["Priority Changed"] = ""

    # Precompute domain tags for TMS rows
    tms["__tags__"] = tms["__full_text__"].str.lower().apply(extract_domain_tags)

    # Progress
    progress = st.progress(0, text="🔄 Building candidate lists…")

    # Build top-K candidates per bug (with optional domain filtering)
    all_edges = []   # list of (score, bug_idx, tms_idx)
    tms_texts = tms["__full_text__"].tolist()
    n_bugs = len(jira)
    for i, bug_text in enumerate(jira["__bug_text__"].tolist()):
        bt = bug_text if isinstance(bug_text, str) else ""
        if not bt.strip():
            progress.progress(int((i+1)/max(1,n_bugs)*100), text="🔄 Building candidate lists…")
            continue

        cand_indices = list(range(len(tms)))
        if use_domain_filter:
            bug_tags = extract_domain_tags(bt)
            if bug_tags:
                mask = [len(tms["__tags__"].iloc[j] & bug_tags) > 0 for j in range(len(tms))]
                filtered = [idx for idx, ok in enumerate(mask) if ok]
                if filtered:
                    cand_indices = filtered

        # Prepare the subset list for matching
        sub_texts = [tms_texts[j] for j in cand_indices]
        # top-k by token_set_ratio for robustness
        matches = process.extract(bt, sub_texts, scorer=fuzz.token_set_ratio, limit=topk)
        # matches: list of (choice_text, score, sub_index)
        for _, score, sub_idx in matches:
            if score >= threshold:
                real_idx = cand_indices[sub_idx]
                all_edges.append((score, i, real_idx))

        progress.progress(int((i+1)/max(1,n_bugs)*100), text="🔄 Building candidate lists…")

    # Greedy global one-to-one assignment (no repeated TMS IDs)
    progress.progress(100, text="✅ Candidates built. Assigning globally…")
    all_edges.sort(reverse=True)  # highest score first
    assigned_bug = set()
    assigned_tms = set()
    chosen = {}  # bug_idx -> (tms_idx, score)

    for score, b_idx, t_idx in all_edges:
        if b_idx in assigned_bug or t_idx in assigned_tms:
            continue
        assigned_bug.add(b_idx)
        assigned_tms.add(t_idx)
        chosen[b_idx] = (t_idx, score)

    # Fill output columns
    for b_idx in range(len(jira)):
        text = jira.at[b_idx, "__bug_text__"]
        # Platform (first match wins)
        jira.at[b_idx, "Platform"] = platform_from_text(text)
        # Components (multi)
        jira.at[b_idx, "Component"] = components_from_text(text)

        if b_idx in chosen:
            t_idx, score = chosen[b_idx]
            jira.at[b_idx, "TMS ID"] = tms.at[t_idx, ID_COL]
            jira.at[b_idx, "Similarity Score"] = float(score)
            # Fill TMS folder if available
            if "Folder" in tms.columns:
                jira.at[b_idx, "TMS Folder Name"] = tms.at[t_idx, "Folder"]

    progress.empty()

    # Final ordering and preview
    desired_order = [
        "Issue Type", "Issue key", "Summary", "Assignee", "Reporter",
        "Priority", "Status", "Review Assignee", "Review Comments",
        "TMS ID", "Component", "Platform", "TMS Folder Name",
        "Priority Changed", "Similarity Score"
    ]
    output_df = jira[[c for c in desired_order if c in jira.columns]].copy()

    st.success("✅ Mapping complete!")
    mapped_count = int((output_df["TMS ID"].astype(str).str.len() > 0).sum()) if "TMS ID" in output_df.columns else 0
    total_count  = len(output_df)
    st.info(f"Mapped {mapped_count} of {total_count} bugs • one-to-one mapping enforced (no repeated TMS IDs).")

    st.subheader("Preview of Mapped Data")
    st.dataframe(output_df.head(50), use_container_width=True)

    # Download
    outbuf = BytesIO()
    with pd.ExcelWriter(outbuf, engine="xlsxwriter") as writer:
        output_df.to_excel(writer, index=False, sheet_name="Mapped")
    st.download_button(
        "📥 Download Result",
        outbuf.getvalue(),
        file_name="Jira_C_List_TMS_Final.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
