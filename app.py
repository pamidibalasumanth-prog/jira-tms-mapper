import re
import time
import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from rapidfuzz import fuzz, process
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =====================================
# Streamlit UI
# =====================================
st.set_page_config(page_title="Jira ↔ TMS Mapper", page_icon="🧭", layout="wide")
st.title("Jira ↔ TMS Mapper")
st.caption("Deterministic multi-stage matching • Developed by Pamidi Bala Sumanth")

# ✅ API key field (not used in this deterministic version, but available for future embedding-based logic)
api_key = st.text_input("🔑 OpenAI API Key (optional)", type="password")

jira_file = st.file_uploader("📂 Upload Jira C List (CSV/XLSX)", type=["csv", "xlsx"])
tms_file  = st.file_uploader("📂 Upload TMS Export (Excel)", type=["xlsx"])

colA, colB, colC = st.columns(3)
with colA:
    fuzzy_threshold = st.slider("Fuzzy threshold", 50, 100, 82, 1,
                                help="Min RapidFuzz token_set_ratio to accept fuzzy match (0-100)")
with colB:
    tfidf_threshold = st.slider("TF-IDF cosine threshold", 0.0, 1.0, 0.22, 0.01,
                                help="Min cosine similarity (TF-IDF) for acceptance")
with colC:
    max_candidates = st.number_input("Max candidate pool size", min_value=50, max_value=2000, value=400, step=50,
                                     help="Restrict matching to this many likely TMS cases per bug")

# =====================================
# Rules (Platform = single; Components = multi)
# =====================================
PLATFORM_RULES = [
    ("salesforce", "Salesforce"),
    ("sap",        "SAP"),
    ("android",    "Mobile"),
    ("ios",        "Mobile"),
    ("mobile",     "Mobile"),
    ("desktop",    "Desktop"),
    ("api",        "API"),
    ("endpoint",   "API"),
    ("rest",       "API"),
    ("browser",    "Web"),
    ("web",        "Web"),
]

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
    "windows": "Windows - .NET",  # heuristic
    "java": "Windows - Java",     # heuristic
}

# =====================================
# Helpers
# =====================================
def clean_text(s: str) -> str:
    if not isinstance(s, str): return ""
    s = re.sub(r"\[.*?\]", " ", s)        # remove [tags]
    s = re.sub(r"\s+", " ", s)
    return s.strip().lower()

def platform_from_text(text: str) -> str:
    t = text.lower()
    for kw, plat in PLATFORM_RULES:
        if kw in t:
            return plat
    return ""

def components_from_text(text: str) -> str:
    t = text.lower()
    comps = []
    for kw, comp in COMPONENT_RULES.items():
        if kw in t:
            comps.append(comp)
    return ", ".join(sorted(set(comps))) if comps else ""

def extract_issue_keys(text: str) -> set:
    """Find Jira-like keys ABC-123 in a text field if present."""
    if not isinstance(text, str): return set()
    return set(re.findall(r"\b[A-Z][A-Z0-9]+-\d+\b", text))

def shortlist_candidates_for_bug(bug_text: str, tms_df: pd.DataFrame, cap: int) -> pd.Index:
    """
    Narrow TMS to likely matches using:
      - Folder/Labels keyword overlap from bug
      - Fuzzy against Title to rank
    """
    tags = set()
    for k in ["salesforce","sap","recorder","api","terminal","agent","integration",
              "billing","license","nlp","tunnel","execution","editor","project",
              "authoring","test plan","co-pilot","documentation","autonomous","heal",
              "atto","ai","windows","java","mobile","android","ios","desktop","web","browser"]:
        if k in bug_text:
            tags.add(k)

    filt = tms_df.index
    if tags:
        mask = tms_df["__folder_labels__"].str.contains("|".join([re.escape(t) for t in tags]), case=False, na=False)
        sub = tms_df[mask]
        if len(sub) >= 1:
            filt = sub.index

    if len(filt) > cap:
        choices = tms_df.loc[filt, "__title__"].tolist()
        ranked = process.extract(bug_text, choices, scorer=fuzz.token_set_ratio, limit=cap)
        sel_idxs = [filt[idx] for _,_,idx in ranked]
        return pd.Index(sel_idxs)

    return pd.Index(filt)

# =====================================
# Main
# =====================================
if jira_file and tms_file:
    jira = pd.read_csv(jira_file) if jira_file.name.endswith(".csv") else pd.read_excel(jira_file)
    if "Issue key" not in jira.columns or "Summary" not in jira.columns:
        st.error("Jira file must include 'Issue key' and 'Summary' columns.")
        st.stop()

    jira["_summary_"] = jira["Summary"].fillna("").astype(str)
    jira["_desc_"]    = jira.get("Description", "").fillna("").astype(str)
    jira["__bug_text__"] = (jira["_summary_"] + " " + jira["_desc_"]).apply(clean_text)

    try:
        tms = pd.read_excel(tms_file, sheet_name="Test Cases")
    except Exception:
        tms = pd.read_excel(tms_file, sheet_name=0)

    # Detect columns
    TMS_ID = [c for c in tms.columns if "id" in c.lower()][0]
    TMS_TITLE = [c for c in tms.columns if "title" in c.lower() or "name" in c.lower()][0]
    TMS_DESC  = "Description" if "Description" in tms.columns else None
    TMS_STEPS = "Steps" if "Steps" in tms.columns else None
    TMS_EXP   = "Expected Results" if "Expected Results" in tms.columns else None
    TMS_FOLDER= "Folder" if "Folder" in tms.columns else None
    TMS_LABEL = "Labels" if "Labels" in tms.columns else None
    TMS_JIRA  = next((c for c in tms.columns if "jira" in c.lower() and "id" in c.lower()), None)

    tms["__title__"] = tms[TMS_TITLE].fillna("").astype(str)
    tms["__desc__"]  = tms[TMS_DESC ].fillna("").astype(str) if TMS_DESC  else ""
    tms["__steps__"] = tms[TMS_STEPS].fillna("").astype(str) if TMS_STEPS else ""
    tms["__exp__"]   = tms[TMS_EXP  ].fillna("").astype(str) if TMS_EXP   else ""
    tms["__folder__"]= tms[TMS_FOLDER].fillna("").astype(str) if TMS_FOLDER else ""
    tms["__labels__"]= tms[TMS_LABEL ].fillna("").astype(str) if TMS_LABEL  else ""
    tms["__folder_labels__"] = (tms["__folder__"] + " " + tms["__labels__"]).str.lower()
    tms["__full_text__"] = (tms["__title__"] + " " + tms["__desc__"] + " " + tms["__steps__"] + " " + tms["__exp__"]).apply(clean_text)

    # Jira → TMS map
    results = []
    progress = st.progress(0.0, text="🔎 Mapping bugs to TMS…")

    for idx, bug in jira.iterrows():
        issue_key = str(bug["Issue key"])
        bug_sum   = bug["_summary_"]
        bug_text  = bug["__bug_text__"]

        platform  = platform_from_text(bug_text)
        components = components_from_text(bug_text)
        picked_tms_id = None

        # 1) If Jira Ticket ID exists in TMS
        if TMS_JIRA and issue_key in str(tms[TMS_JIRA].values):
            matches = tms[tms[TMS_JIRA].astype(str).str.contains(issue_key, na=False)]
            if not matches.empty:
                picked_tms_id = str(matches.iloc[0][TMS_ID])

        # 2) Title containment
        if picked_tms_id is None:
            pool = shortlist_candidates_for_bug(bug_text, tms, cap=max_candidates)
            bug_norm = clean_text(bug_sum)
            for j in pool:
                t = tms.at[j, "__title__"].lower()
                if bug_norm and (bug_norm in t or t in bug_norm):
                    if fuzz.token_set_ratio(bug_sum, t) >= fuzzy_threshold:
                        picked_tms_id = str(tms.at[j, TMS_ID])
                        break

        # 3) Fuzzy
        if picked_tms_id is None:
            pool = shortlist_candidates_for_bug(bug_text, tms, cap=max_candidates)
            choices = tms.loc[pool, "__full_text__"].tolist()
            match = process.extractOne(bug_text, choices, scorer=fuzz.token_set_ratio)
            if match and match[1] >= fuzzy_threshold:
                picked_tms_id = str(tms.at[pool[match[2]], TMS_ID])

        # 4) TF-IDF
        if picked_tms_id is None:
            pool = shortlist_candidates_for_bug(bug_text, tms, cap=max_candidates)
            tfidf = TfidfVectorizer(min_df=1, ngram_range=(1,2))
            corpus = [bug_text] + tms.loc[pool, "__full_text__"].tolist()
            try:
                X = tfidf.fit_transform(corpus)
                sims = cosine_similarity(X[0:1], X[1:]).ravel()
                best_pos = int(np.argmax(sims))
                if sims[best_pos] >= tfidf_threshold:
                    picked_tms_id = str(tms.at[pool[best_pos], TMS_ID])
            except ValueError:
                pass

        if picked_tms_id is None:
            picked_tms_id = "Test case needs to be added – no match found"

        results.append({
            "Issue key": issue_key,
            "Summary": bug_sum,
            "TMS ID": picked_tms_id,
            "Component": components,
            "Platform": platform
        })

        if (idx + 1) % 5 == 0 or idx == len(jira) - 1:
            progress.progress((idx + 1) / len(jira), text=f"🔎 Mapping… {idx+1}/{len(jira)}")

    progress.empty()

    final_df = pd.DataFrame(results, columns=["Issue key", "Summary", "TMS ID", "Component", "Platform"])
    mapped_cnt = (final_df["TMS ID"] != "Test case needs to be added – no match found").sum()
    st.success(f"✅ Done • {mapped_cnt}/{len(final_df)} mapped • {len(final_df)-mapped_cnt} need new test cases")

    st.dataframe(final_df, use_container_width=True, height=720)

    out = BytesIO()
    with pd.ExcelWriter(out, engine="xlsxwriter") as writer:
        final_df.to_excel(writer, index=False, sheet_name="Mapped")
    st.download_button(
        "📥 Download Excel (Mapped)",
        data=out.getvalue(),
        file_name="Jira_C_List_TMS_Final.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
