# app.py
import os
import re
import json
import time
import hashlib
import random
from io import BytesIO
from typing import List, Tuple

import streamlit as st
import pandas as pd
import numpy as np
from rapidfuzz import fuzz, process
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI

# ================================
# Streamlit UI
# ================================
st.set_page_config(page_title="Jira ↔ TMS Mapper (Hybrid)", page_icon="🧭", layout="wide")
st.title("Jira ↔ TMS Mapper (Hybrid)")
st.caption("Fast candidate filtering → semantic rerank (cached) → optional LLM review • Developed by Pamidi Bala Sumanth")

with st.expander("🔐 API & Settings", expanded=True):
    api_key = st.text_input("OpenAI API Key (required for embeddings & LLM review)", type="password")
    use_llm = st.toggle("Use LLM Reviewer (optional, slower, more precise)", value=False,
                        help="If ON, best match will be validated by an LLM. If rejected → 'Test case needs to be added'.")

left, mid, right = st.columns(3)
with left:
    fuzzy_threshold = st.slider("Fuzzy threshold (0–100)", 50, 100, 82, 1)
with mid:
    tfidf_threshold = st.slider("TF-IDF cosine threshold", 0.0, 1.0, 0.22, 0.01)
with right:
    candidate_cap = st.number_input("Max candidates per bug", 50, 1000, 300, 50)

sim_threshold = st.slider("Semantic similarity threshold (0–1)", 0.0, 1.0, 0.60, 0.01,
                          help="Min cosine for semantic rerank to accept a match.")

jira_file = st.file_uploader("📂 Upload Jira C List (CSV/XLSX)", type=["csv", "xlsx"])
tms_file  = st.file_uploader("📂 Upload TMS Export (Excel)", type=["xlsx"])

# ================================
# Rules (Platform single, Component multi)
# ================================
PLATFORM_RULES: List[Tuple[str, str]] = [
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

TAG_KEYS = [
    "salesforce","sap","recorder","api","terminal","agent","integration",
    "billing","license","nlp","tunnel","execution","editor","project",
    "authoring","test plan","co-pilot","documentation","autonomous","heal",
    "atto","ai","windows","java","mobile","android","ios","desktop","web","browser"
]

# ================================
# Helpers
# ================================
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
    if not isinstance(text, str): return set()
    return set(re.findall(r"\b[A-Z][A-Z0-9]+-\d+\b", text))

def shortlist_candidates_for_bug(bug_text: str, tms_df: pd.DataFrame, cap: int) -> pd.Index:
    # tag filter using Folder/Labels
    tags = {k for k in TAG_KEYS if k in bug_text}
    filt = tms_df.index
    if tags:
        mask = tms_df["__folder_labels__"].str.contains("|".join([re.escape(t) for t in tags]), case=False, na=False)
        sub = tms_df[mask]
        if len(sub) >= 1:
            filt = sub.index

    # cap with fuzzy by Title
    if len(filt) > cap:
        choices = tms_df.loc[filt, "__title__"].tolist()
        ranked = process.extract(bug_text, choices, scorer=fuzz.token_set_ratio, limit=cap)
        sel = [filt[i2] for _,_,i2 in ranked]
        return pd.Index(sel)

    return pd.Index(filt)

def file_hash_of_df(df: pd.DataFrame, cols: List[str]) -> str:
    h = hashlib.md5()
    for c in cols:
        vals = df[c].fillna("").astype(str).tolist()
        for v in vals:
            h.update(v.encode("utf-8", errors="ignore"))
            h.update(b"|")
    return h.hexdigest()

def ensure_api(client_needed: bool):
    if client_needed and not api_key:
        st.error("OpenAI API key is required for embeddings/LLM review. Please enter it in the header.")
        st.stop()

# ================================
# Embeddings (cached to disk)
# ================================
def embed_texts_batched(client: OpenAI, texts: List[str], model="text-embedding-3-small", batch_size=100) -> np.ndarray:
    vectors = []
    prog = st.progress(0.0, text="🔎 Embedding (batched)…")
    total = len(texts)
    for i in range(0, total, batch_size):
        batch = [t if isinstance(t, str) and t.strip() else " " for t in texts[i:i+batch_size]]
        success = False
        while not success:
            try:
                resp = client.embeddings.create(model=model, input=batch)
                vectors.extend([np.array(d.embedding, dtype=np.float32) for d in resp.data])
                success = True
            except Exception as e:
                # backoff on rate limits/transient errors
                time.sleep(random.uniform(4, 10))
        prog.progress(min(0.999, (i+batch_size)/max(1,total)))
    prog.progress(1.0, text="✅ Embeddings ready")
    return np.vstack(vectors) if vectors else np.zeros((0, 1536), dtype=np.float32)

def load_or_create_tms_embeddings(client: OpenAI, tms_df: pd.DataFrame, text_col: str, cache_dir: str = ".cache_emb") -> np.ndarray:
    os.makedirs(cache_dir, exist_ok=True)
    h = file_hash_of_df(tms_df, [text_col])
    path = os.path.join(cache_dir, f"tms_{h}.npy")
    if os.path.exists(path):
        try:
            embs = np.load(path)
            return embs
        except Exception:
            pass
    texts = tms_df[text_col].tolist()
    embs = embed_texts_batched(client, texts, model="text-embedding-3-small", batch_size=100)
    np.save(path, embs)
    return embs

# ================================
# Optional LLM review
# ================================
def llm_review_match(client: OpenAI, bug_text: str, case_text: str) -> Tuple[bool, str]:
    """
    Ask LLM: Does this test case truly test the bug?
    Returns (accepted_bool, short_comment)
    """
    try:
        msg = [
            {"role": "system", "content": "You are a QA lead. Answer with a short, concrete judgement."},
            {"role": "user", "content": (
                "Bug report:\n"
                f"{bug_text}\n\n"
                "Candidate test case:\n"
                f"{case_text}\n\n"
                "Question: Does this test case correctly cover and validate the behavior/regression described in the bug? "
                "Answer strictly as JSON with keys: accept (true/false), reason (<=15 words)."
            )}
        ]
        # gpt-4o-mini is a good balance; change if your org prefers different model
        resp = client.chat.completions.create(model="gpt-4o-mini", messages=msg, temperature=0)
        content = resp.choices[0].message.content.strip()
        # try parse JSON
        data = json.loads(content) if content.startswith("{") else {}
        accept = bool(data.get("accept", False))
        reason = str(data.get("reason", "")).strip()
        return accept, reason or ("Looks aligned" if accept else "Not aligned")
    except Exception:
        # on any error, don't block; accept as-is with neutral reason
        return True, "LLM unavailable, auto-accepted"
        
# ================================
# MAIN
# ================================
if jira_file and tms_file:
    # ---- Load Jira
    jira = pd.read_csv(jira_file) if jira_file.name.endswith(".csv") else pd.read_excel(jira_file)
    if "Issue key" not in jira.columns or "Summary" not in jira.columns:
        st.error("Jira file must include 'Issue key' and 'Summary' columns.")
        st.stop()

    jira["_summary_"] = jira["Summary"].fillna("").astype(str)
    jira["_desc_"]    = jira.get("Description", "").fillna("").astype(str)
    jira["__bug_text__"] = (jira["_summary_"] + " " + jira["_desc_"]).apply(clean_text)

    # ---- Load TMS
    try:
        tms = pd.read_excel(tms_file, sheet_name="Test Cases")
    except Exception:
        tms = pd.read_excel(tms_file, sheet_name=0)

    # Detect essential columns
    tms_id_col = [c for c in tms.columns if "id" in c.lower()]
    if not tms_id_col:
        st.error("TMS sheet must include an 'ID' column.")
        st.stop()
    TMS_ID = tms_id_col[0]

    title_cols = [c for c in tms.columns if "title" in c.lower() or "name" in c.lower()]
    if not title_cols:
        st.error("TMS sheet must include a 'Title' or 'Name' column.")
        st.stop()
    TMS_TITLE = title_cols[0]

    TMS_DESC  = "Description" if "Description" in tms.columns else None
    TMS_STEPS = "Steps" if "Steps" in tms.columns else None
    TMS_EXP   = "Expected Results" if "Expected Results" in tms.columns else None
    TMS_FOLDER= "Folder" if "Folder" in tms.columns else None
    TMS_LABEL = "Labels" if "Labels" in tms.columns else None
    TMS_JIRA  = next((c for c in tms.columns if "jira" in c.lower() and "id" in c.lower()), None)

    # Precompute normalized fields for TMS
    tms["__title__"]  = tms[TMS_TITLE].fillna("").astype(str)
    tms["__desc__"]   = tms[TMS_DESC ].fillna("").astype(str) if TMS_DESC  else ""
    tms["__steps__"]  = tms[TMS_STEPS].fillna("").astype(str) if TMS_STEPS else ""
    tms["__exp__"]    = tms[TMS_EXP  ].fillna("").astype(str) if TMS_EXP   else ""
    tms["__folder__"] = tms[TMS_FOLDER].fillna("").astype(str) if TMS_FOLDER else ""
    tms["__labels__"] = tms[TMS_LABEL ].fillna("").astype(str) if TMS_LABEL  else ""
    tms["__folder_labels__"] = (tms["__folder__"] + " " + tms["__labels__"]).str.lower()

    tms["__full_text__"] = (
        tms["__title__"] + " " + tms["__desc__"] + " " + tms["__steps__"] + " " + tms["__exp__"]
    ).apply(clean_text)

    # ---- Optional exact join by Jira Ticket ID (ground truth)
    tms_issuekey_to_idxs = {}
    if TMS_JIRA:
        for i, v in tms[TMS_JIRA].fillna("").astype(str).items():
            for k in extract_issue_keys(v):
                tms_issuekey_to_idxs.setdefault(k, []).append(i)

    # ---- Ensure API if embeddings/LLM needed
    need_api = True  # embeddings are required for semantic rerank
    ensure_api(need_api)
    client = OpenAI(api_key=api_key)

    # ---- Cache & compute embeddings for TMS (once per content)
    st.info("⏳ Preparing embeddings cache for TMS cases…")
    tms_emb = load_or_create_tms_embeddings(client, tms, "__full_text__")

    # ---- For each bug: candidate shortlist → semantic rerank → (optional LLM review)
    results = []
    progress = st.progress(0.0, text="🔗 Mapping bugs to TMS…")

    # Pre-embed Jira bugs once
    st.info("⏳ Embedding Jira bugs…")
    jira_emb = embed_texts_batched(client, jira["__bug_text__"].tolist(), model="text-embedding-3-small", batch_size=100)

    for i, bug in jira.iterrows():
        issue_key = str(bug["Issue key"])
        bug_sum   = bug["_summary_"]
        bug_text  = bug["__bug_text__"]
        platform  = platform_from_text(bug_text)
        comps     = components_from_text(bug_text)

        chosen_id = None
        review_comment = ""

        # 0) Exact link via TMS Jira Ticket ID column
        if TMS_JIRA and issue_key in tms_issuekey_to_idxs:
            idxs = tms_issuekey_to_idxs[issue_key]
            if len(idxs) == 1:
                chosen_id = str(tms.at[idxs[0], TMS_ID])
            else:
                # If multiple refer same issue, pick by fuzzy title vs bug summary
                bestj, bests = None, -1
                for j in idxs:
                    s = fuzz.token_set_ratio(bug_sum, tms.at[j, "__title__"])
                    if s > bests:
                        bests, bestj = s, j
                chosen_id = str(tms.at[bestj, TMS_ID])

        # 1) Candidate shortlist
        pool = shortlist_candidates_for_bug(bug_text, tms, cap=candidate_cap)

        # 2) Quick checks (title containment then fuzzy)
        if chosen_id is None:
            bug_norm = clean_text(bug_sum)
            contain_hits = []
            for j in pool:
                t = tms.at[j, "__title__"].lower()
                if bug_norm and (bug_norm in t or t in bug_norm):
                    contain_hits.append((j, fuzz.token_set_ratio(bug_sum, t)))
            if contain_hits:
                contain_hits.sort(key=lambda x: x[1], reverse=True)
                if contain_hits[0][1] >= fuzzy_threshold:
                    chosen_id = str(tms.at[contain_hits[0][0], TMS_ID])

        if chosen_id is None:
            choices = tms.loc[pool, "__full_text__"].tolist()
            match = process.extractOne(bug_text, choices, scorer=fuzz.token_set_ratio)
            if match and match[1] >= fuzzy_threshold:
                chosen_id = str(tms.at[pool[match[2]], TMS_ID])

        # 3) TF-IDF cosine within pool
        if chosen_id is None:
            try:
                tfidf = TfidfVectorizer(min_df=1, ngram_range=(1,2))
                corpus = [bug_text] + tms.loc[pool, "__full_text__"].tolist()
                X = tfidf.fit_transform(corpus)
                sims = cosine_similarity(X[0:1], X[1:]).ravel()
                best_pos = int(np.argmax(sims))
                if sims[best_pos] >= tfidf_threshold:
                    chosen_id = str(tms.at[pool[best_pos], TMS_ID])
            except ValueError:
                pass

        # 4) Semantic rerank among pool (final arbiter)
        best_sem_id = None
        best_sem_score = -1.0
        bvec = jira_emb[i]
        for j in pool:
            cvec = tms_emb[j]
            denom = (np.linalg.norm(bvec) * np.linalg.norm(cvec) + 1e-9)
            score = float(np.dot(bvec, cvec) / denom)
            if score > best_sem_score:
                best_sem_score, best_sem_id = score, str(tms.at[j, TMS_ID])

        # accept semantic only if above threshold & clearly better than other signals, otherwise keep earlier choice
        if best_sem_score >= sim_threshold:
            chosen_id = best_sem_id

        # 5) Optional LLM reviewer to confirm top match
        if use_llm and chosen_id and "needs to be added" not in chosen_id.lower():
            jrow = tms[tms[TMS_ID].astype(str) == chosen_id]
            if not jrow.empty:
                case_txt = jrow.iloc[0]["__full_text__"]
                ok, reason = llm_review_match(client, bug_text, case_txt)
                review_comment = reason
                if not ok:
                    chosen_id = "Test case needs to be added – no match found"

        if not chosen_id:
            chosen_id = "Test case needs to be added – no match found"

        results.append({
            "Issue key": issue_key,
            "Summary": bug_sum,
            "TMS ID": chosen_id,
            "Component(s)": comps,
            "Platform": platform,
            "Similarity Score": round(float(best_sem_score), 4) if best_sem_score >= 0 else 0.0,
            "Review Comment": review_comment
        })

        if (i+1) % 5 == 0 or i == len(jira)-1:
            progress.progress((i+1)/len(jira), text=f"🔗 Mapping… {i+1}/{len(jira)}")

    progress.empty()

    final_df = pd.DataFrame(results, columns=[
        "Issue key","Summary","TMS ID","Component(s)","Platform","Similarity Score","Review Comment"
    ])

    mapped_cnt = (final_df["TMS ID"] != "Test case needs to be added – no match found").sum()
    st.success(f"✅ Complete • {mapped_cnt}/{len(final_df)} mapped • {len(final_df)-mapped_cnt} need new test cases")

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
