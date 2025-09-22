import streamlit as st
import pandas as pd
import numpy as np
from openai import OpenAI
from rapidfuzz import fuzz
from io import BytesIO
import time

# -----------------------------------
# Streamlit Page Setup
# -----------------------------------
st.set_page_config(page_title="Jira ↔ TMS Mapper", layout="wide")
st.title("Jira ↔ TMS Mapper")
st.caption("Developed by Pamidi Bala Sumanth")

# -----------------------------------
# Step 1: API Key Input
# -----------------------------------
api_key = st.text_input("🔑 Enter your OpenAI API key", type="password")

if not api_key:
    st.warning("⚠️ Please enter your OpenAI API key above to continue.")
    st.stop()

client = OpenAI(api_key=api_key)

# -----------------------------------
# Helper Functions
# -----------------------------------
def get_embeddings_batched(texts, client, model="text-embedding-3-small", batch_size=100):
    """Batch embedding with progress bar"""
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
            st.error(f"Embedding error at batch {i//batch_size}: {e}")
            for _ in batch:
                embeddings.append([0.0] * 1536)

        # Update progress
        done = min(i+batch_size, total)
        progress.progress(done/total)
        status.text(f"Processed {done}/{total} rows...")

        time.sleep(0.2)  # slight delay for rate limits

    progress.empty()
    status.text("✅ Embeddings ready!")
    return np.array(embeddings)


def cosine_similarity(vec1, vec2):
    v1, v2 = np.array(vec1), np.array(vec2)
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10))


def hybrid_similarity(text1, text2, emb1, emb2, alpha=0.7):
    """Combine embeddings (semantic) + fuzzy ratio (literal match)"""
    cos_sim = cosine_similarity(emb1, emb2)
    fuzz_score = fuzz.token_sort_ratio(text1, text2) / 100
    return alpha * cos_sim + (1 - alpha) * fuzz_score


# -----------------------------------
# File Uploads
# -----------------------------------
jira_file = st.file_uploader("📂 Upload Jira C List (CSV/XLSX)", type=["csv", "xlsx"])
tms_file = st.file_uploader("📂 Upload TMS Export (Excel)", type=["xlsx"])
threshold = st.slider("🎯 Similarity threshold", 0.0, 1.0, 0.65, 0.01)

# -----------------------------------
# Processing
# -----------------------------------
if jira_file and tms_file:
    # Load Jira
    if jira_file.name.endswith(".csv"):
        jira = pd.read_csv(jira_file)
    else:
        jira = pd.read_excel(jira_file)

    # Load TMS
    tms = pd.read_excel(tms_file)

    # Prepare text fields
    jira["__bug_text__"] = jira["Summary"].fillna("") + " " + jira.get("Description", "").fillna("")
    tms["__case_text__"] = tms["Title"].fillna("") + " " + tms["Description"].fillna("")

    # Generate embeddings
    st.info("⏳ Generating embeddings for Jira bugs...")
    jira_embs = get_embeddings_batched(jira["__bug_text__"].tolist(), client)

    st.info("⏳ Generating embeddings for TMS cases (this may take longer)...")
    tms_embs = get_embeddings_batched(tms["__case_text__"].tolist(), client)

    # Perform matching
    mappings = []
    progress = st.progress(0)
    status = st.empty()

    for i, bug in jira.iterrows():
        bug_text = bug["__bug_text__"]
        bug_emb = jira_embs[i]

        best_score, best_case = -1, None

        for j, case in tms.iterrows():
            score = hybrid_similarity(bug_text, case["__case_text__"], bug_emb, tms_embs[j])
            if score > best_score:
                best_score, best_case = score, case

        if best_score >= threshold and best_case is not None:
            mappings.append({
                **bug.to_dict(),
                "TMS ID": best_case["ID"],
                "TMS Folder Name": best_case.get("Folder", ""),
                "Similarity Score": round(best_score, 4),
                "Component": bug.get("Component", ""),
                "Platform": bug.get("Platform", "")
            })
        else:
            mappings.append({
                **bug.to_dict(),
                "TMS ID": "❌ Test case needs to be added",
                "TMS Folder Name": "New",
                "Similarity Score": round(best_score, 4),
                "Component": bug.get("Component", ""),
                "Platform": bug.get("Platform", "")
            })

        progress.progress((i+1)/len(jira))
        status.text(f"Matched {i+1}/{len(jira)} Jira bugs")

    progress.empty()
    status.text("✅ Mapping complete!")

    mapped_df = pd.DataFrame(mappings)

    # Show all rows
    st.write(mapped_df.to_html(escape=False, index=False), unsafe_allow_html=True)

    # Download Excel
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        mapped_df.to_excel(writer, index=False, sheet_name="Mapped")
    st.download_button(
        "📥 Download Excel",
        data=output.getvalue(),
        file_name="Jira_C_List_TMS_Final.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
