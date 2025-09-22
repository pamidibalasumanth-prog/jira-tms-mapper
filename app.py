import streamlit as st
import pandas as pd
import numpy as np
from openai import OpenAI
from rapidfuzz import fuzz
from io import BytesIO
import time

# --------------------------
# Streamlit Page Setup
# --------------------------
st.set_page_config(page_title="Jira ↔ TMS Mapper", layout="wide")
st.title("Jira ↔ TMS Mapper")
st.caption("Developed by Pamidi Bala Sumanth")

# --------------------------
# API Key Input
# --------------------------
api_key = st.text_input("🔑 Enter your OpenAI API key", type="password")

if not api_key:
    st.warning("⚠️ Please enter your OpenAI API key above to continue.")
    st.stop()

client = OpenAI(api_key=api_key)

# --------------------------
# Helper Functions
# --------------------------
def get_embeddings_batched(texts, client, model="text-embedding-3-small", batch_size=100):
    embeddings = []
    progress = st.progress(0)
    status = st.empty()
    total = len(texts)

    for i in range(0, total, batch_size):
        batch = texts[i:i+batch_size]
        try:
            response = client.embeddings.create(model=model, input=batch)
            for emb in response.data:
                embeddings.append(emb.embedding)
        except Exception as e:
            st.error(f"Embedding error: {e}")
            embeddings.extend([[0.0]*1536]*len(batch))
        done = min(i+batch_size, total)
        progress.progress(done/total)
        status.text(f"Processed {done}/{total}")
        time.sleep(0.2)
    progress.empty()
    status.text("✅ Embeddings ready!")
    return np.array(embeddings)

def cosine_similarity(v1, v2):
    v1, v2 = np.array(v1), np.array(v2)
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10))

def assign_component(summary):
    summary = summary.lower()
    if "salesforce" in summary: return "SalesForce"
    if "sap" in summary: return "SAP"
    if "recorder" in summary: return "Recorder (Browser)"
    if "user" in summary: return "User Management"
    return "General"

def assign_platform(summary):
    summary = summary.lower()
    if "android" in summary: return "Android"
    if "ios" in summary: return "iOS"
    if "windows" in summary: return "Windows"
    if "mac" in summary: return "Mac"
    return "Web"

# --------------------------
# File Uploads
# --------------------------
jira_file = st.file_uploader("📂 Upload Jira C List (CSV/XLSX)", type=["csv", "xlsx"])
tms_file = st.file_uploader("📂 Upload TMS Export (Excel)", type=["xlsx"])
threshold = st.slider("🎯 Similarity threshold", 0.0, 1.0, 0.65, 0.01)

# --------------------------
# Processing
# --------------------------
if jira_file and tms_file:
    if jira_file.name.endswith(".csv"):
        jira = pd.read_csv(jira_file)
    else:
        jira = pd.read_excel(jira_file)
    tms = pd.read_excel(tms_file)

    jira["__bug_text__"] = jira["Summary"].fillna("")
    tms["__case_text__"] = tms["Title"].fillna("") + " " + tms["Description"].fillna("")

    st.info("⏳ Embedding Jira bugs...")
    jira_embs = get_embeddings_batched(jira["__bug_text__"].tolist(), client)

    st.info("⏳ Embedding TMS cases...")
    tms_embs = get_embeddings_batched(tms["__case_text__"].tolist(), client)

    results = []
    for i, bug in jira.iterrows():
        bug_text = bug["__bug_text__"]
        bug_emb = jira_embs[i]

        best_score, best_case = -1, None
        for j, case in tms.iterrows():
            score = cosine_similarity(bug_emb, tms_embs[j])
            if score > best_score:
                best_score, best_case = score, case

        if best_score >= threshold and best_case is not None:
            tms_id = best_case["ID"]
        else:
            tms_id = "❌ Test case needs to be added"

        results.append({
            "Issue key": bug["Issue key"],
            "Summary": bug["Summary"],
            "TMS ID": tms_id,
            "Component": assign_component(bug["Summary"]),
            "Platform": assign_platform(bug["Summary"])
        })

    final_df = pd.DataFrame(results)

    st.success("✅ Mapping complete!")
    st.dataframe(final_df, use_container_width=True)

    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        final_df.to_excel(writer, index=False, sheet_name="Mapped")
    st.download_button(
        "📥 Download Excel",
        data=output.getvalue(),
        file_name="Jira_C_List_TMS_Final.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
