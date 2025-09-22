import os
import re
import time
import hashlib
import random
import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import fuzz, process

# =========================
# Streamlit Setup
# =========================
st.set_page_config(page_title="Jira ↔ TMS Mapper", page_icon="🧭", layout="wide")
st.title("Jira ↔ TMS Mapper")
st.caption("Developed by Pamidi Bala Sumanth • precise mapping + auto-fallback")

# =========================
# Controls
# =========================
api_key = st.text_input("🔑 OpenAI API key", type="password")
jira_file = st.file_uploader("📂 Upload Jira C List (CSV/XLSX)", type=["csv", "xlsx"])
tms_file  = st.file_uploader("📂 Upload TMS Export (Excel)", type=["xlsx"])

col1, col2, col3 = st.columns(3)
with col1:
    sim_threshold = st.slider("Similarity threshold", 0.0, 1.0, 0.60, 0.01,
                              help="Higher = stricter (fewer matches), Lower = looser (more matches)")
with col2:
    candidate_cap = st.number_input("Candidate pool per bug (after filtering)", 50, 1000, 300, step=50,
                                    help="We only compare against this many likely TMS cases per bug")
with col3:
    alpha = st.slider("Semantic vs Fuzzy weight (α)", 0.0, 1.0, 0.80, 0.05,
                      help="α=1 uses only embeddings; α=0 uses only fuzzy string match")

if not api_key:
    st.info("Enter API key to begin.")
    st.stop()

client = OpenAI(api_key=api_key)

# =========================
# Hardcoded rules (Platform = single; Component = multi)
# =========================
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

# =========================
# Helpers
# =========================
def clean_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = re.sub(r"\[.*?\]", " ", s)  # remove [Prod], [Request], ...
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

def extract_tags(text: str) -> set:
    t = text.lower()
    keys = [
        "salesforce","sap","recorder","api","terminal","agent","integration",
        "billing","license","nlp","tunnel","execution","editor","project",
        "authoring","test plan","co-pilot","documentation","autonomous","heal",
        "atto","ai","windows","java","mobile","android","ios","desktop","web","browser"
    ]
    return {k for k in keys if k in t}

def hash_series_text(series: pd.Series) -> str:
    m = hashlib.md5()
    for v in series.values:
        m.update(str(v).encode("utf-8", errors="ignore"))
        m.update(b"|")
    return m.hexdigest()

@st.cache_data(show_spinner=False)
def embed_texts_cached(texts: list, model="text-embedding-3-small", batch_size=100):
    """Return np.ndarray of shape (N, 1536). Cached by content hash."""
    content_key = hashlib.md5("|".join(texts).encode("utf-8", errors="ignore")).hexdigest()
    # return cached value if exists
    cached = st.session_state.get(f"emb_{content_key}")
    if cached is not None:
        return cached

    vectors = []
    prog = st.progress(0.0, text="🔎 Embedding texts (batched)…")
    total = len(texts)
    for i in range(0, total, batch_size):
        batch = texts[i:i+batch_size]
        # replace empties with short placeholder to keep alignment
        safe_batch = [b if isinstance(b, str) and b.strip() else " " for b in batch]
        success = False
        while not success:
            try:
                resp = client.embeddings.create(model=model, input=safe_batch)
                vectors.extend([np.array(d.embedding, dtype=np.float32) for d in resp.data])
                success = True
            except Exception:
                time.sleep(random.uniform(4, 10))
        prog.progress(min(0.999, (i+batch_size)/max(1,total)))
    prog.progress(1.0, text="✅ Embeddings ready")
    out = np.vstack(vectors)
    st.session_state[f"emb_{content_key}"] = out
    return out

def hybrid_score(btxt: str, ctxt: str, bvec: np.ndarray, cvec: np.ndarray, alpha: float) -> float:
    # semantic
    cos = float(np.dot(bvec, cvec) / (np.linalg.norm(bvec) * np.linalg.norm(cvec) + 1e-9))
    # literal
    fuzz_score = fuzz.token_set_ratio(btxt, ctxt) / 100.0
    return alpha * cos + (1.0 - alpha) * fuzz_score

# =========================
# Main
# =========================
if jira_file and tms_file:
    # ---- Load Jira
    jira = pd.read_csv(jira_file) if jira_file.name.endswith(".csv") else pd.read_excel(jira_file)
    if "Summary" not in jira.columns or "Issue key" not in jira.columns:
        st.error("Jira file must include 'Issue key' and 'Summary' columns.")
        st.stop()
    jira["__bug_text__"] = (jira["Summary"].fillna("") + " " + jira.get("Description", "").fillna("")).apply(clean_text)
    jira["__tags__"] = jira["__bug_text__"].apply(extract_tags)

    # ---- Load TMS
    try:
        tms = pd.read_excel(tms_file, sheet_name="Test Cases")
    except Exception:
        tms = pd.read_excel(tms_file, sheet_name=0)

    # detect columns
    try:
        tms_id_col = [c for c in tms.columns if "id" in c.lower()][0]
    except IndexError:
        st.error("Could not find a TMS ID column. Ensure your TMS sheet has an 'ID' column.")
        st.stop()
    try:
        tms_title_col = [c for c in tms.columns if "title" in c.lower()][0]
    except IndexError:
        st.error("Could not find a TMS Title column. Ensure your TMS sheet has a 'Title' column.")
        st.stop()

    # Build text + tags for TMS
    tms["__full_text__"] = (
        tms[tms_title_col].fillna("").astype(str) + " " +
        tms.get("Description", "").fillna("").astype(str) + " " +
        tms.get("Steps", "").fillna("").astype(str) + " " +
        tms.get("Expected Results", "").fillna("").astype(str) + " " +
        tms.get("Folder", "").fillna("").astype(str) + " " +
        tms.get("Labels", "").fillna("").astype(str)
    ).apply(clean_text)
    tms["__tags__"] = tms["__full_text__"].apply(extract_tags)

    # ---- Embeddings (with caching)
    st.info("⏳ Embedding TMS cases (cached by file content)…")
    tms_emb = embed_texts_cached(tms["__full_text__"].tolist(), model="text-embedding-3-small", batch_size=100)
    st.info("⏳ Embedding Jira bugs…")
    jira_emb = embed_texts_cached(jira["__bug_text__"].tolist(), model="text-embedding-3-small", batch_size=100)

    # ---- Candidate generation per bug (domain filter → fuzzy shortlist → cap)
    progress_candidates = st.progress(0.0, text="🔍 Selecting candidates…")
    candidates_per_bug = []
    for i, row in jira.iterrows():
        bug_tags = row["__tags__"]
        # domain filter: prefer TMS rows sharing tags
        mask = (tms["__tags__"].apply(lambda s: len(s & bug_tags) > 0))
        idxs = list(tms.index[mask])
        if not idxs:
            # fallback: fuzzy shortlist by Title only (fast)
            choices = tms[tms_title_col].fillna("").tolist()
            picks = process.extract(row["__bug_text__"], choices, scorer=fuzz.token_set_ratio, limit=min(candidate_cap, len(choices)))
            idxs = [p[2] for p in picks]  # indices into choices; matches tms rows order
        else:
            # if still big, cap by fuzzy on full_text
            if len(idxs) > candidate_cap:
                subset = tms["__full_text__"].iloc[idxs].tolist()
                picks = process.extract(row["__bug_text__"], subset, scorer=fuzz.token_set_ratio, limit=candidate_cap)
                idxs = [idxs[p[2]] for p in picks]
        candidates_per_bug.append(idxs)
        progress_candidates.progress((i+1)/len(jira), text=f"🔍 Selecting candidates… {i+1}/{len(jira)}")
    progress_candidates.empty()

    # ---- Score edges for global one-to-one
    st.info("⚖️ Scoring candidates & enforcing one-to-one mapping…")
    edges = []  # (score, bug_idx, tms_idx, best2)
    for i, row in jira.iterrows():
        bug_vec = jira_emb[i]
        bug_txt = row["__bug_text__"]
        plat = platform_from_text(bug_txt)
        comps = components_from_text(bug_txt)
        bug_tags = row["__tags__"]

        scores = []
        for j in candidates_per_bug[i]:
            cvec = tms_emb[j]
            ctxt = tms.at[j, "__full_text__"]

            # base hybrid score
            s = hybrid_score(bug_txt, ctxt, bug_vec, cvec, alpha=alpha)

            # domain boosts
            # tag overlap
            overlap = len(bug_tags & tms.at[j, "__tags__"])
            s += min(0.02 * overlap, 0.08)

            # folder/component hint
            if "Folder" in tms.columns and isinstance(tms.at[j, "Folder"], str):
                if comps:
                    # if any component name appears in folder text, boost
                    if any(c.strip().lower() in tms.at[j, "Folder"].lower() for c in comps.split(",")):
                        s += 0.05

            # platform hint
            if plat and plat.lower() in tms.at[j, "__full_text__"]:
                s += 0.05

            scores.append((s, j))

        if scores:
            scores.sort(reverse=True, key=lambda x: x[0])
            best, best_j = scores[0][0], scores[0][1]
            best2 = scores[1][0] if len(scores) > 1 else 0.0
            edges.append((best, i, best_j, best2))

    # Greedy assignment (highest score first), prevents repeated TMS IDs
    edges.sort(reverse=True, key=lambda x: x[0])
    assigned_bug, assigned_tms, chosen = set(), set(), {}
    for s, bi, tj, s2 in edges:
        if s < sim_threshold:
            continue
        if bi in assigned_bug or tj in assigned_tms:
            continue
        # Optional: ensure uniqueness by checking margin to runner-up
        if s - s2 < 0.03:  # small margin → too ambiguous, skip
            continue
        assigned_bug.add(bi)
        assigned_tms.add(tj)
        chosen[bi] = (tj, s)

    # ---- Build final output (exactly 5 columns)
    rows = []
    for i, row in jira.iterrows():
        bug_txt = row["__bug_text__"]
        plat = platform_from_text(bug_txt)
        comps = components_from_text(bug_txt)

        if i in chosen:
            j = chosen[i][0]
            tms_id = str(tms.at[j, tms_id_col])
        else:
            tms_id = "Test case needs to be added – no match found"

        rows.append({
            "Issue key": row["Issue key"],
            "Summary": row["Summary"],
            "TMS ID": tms_id,
            "Component": comps,      # multiple allowed, comma-separated
            "Platform": plat         # only one
        })

    final_df = pd.DataFrame(rows, columns=["Issue key", "Summary", "TMS ID", "Component", "Platform"])

    # ---- Preview (all rows, clean)
    st.success(f"✅ Done: { (final_df['TMS ID']!='Test case needs to be added – no match found').sum() } mapped • "
               f"{ (final_df['TMS ID']=='Test case needs to be added – no match found').sum() } need new cases")
    st.dataframe(final_df, use_container_width=True, height=720)

    # ---- Download (exact same columns)
    out = BytesIO()
    with pd.ExcelWriter(out, engine="xlsxwriter") as writer:
        final_df.to_excel(writer, index=False, sheet_name="Mapped")
    st.download_button(
        "📥 Download Excel (Mapped)",
        data=out.getvalue(),
        file_name="Jira_C_List_TMS_Final.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
