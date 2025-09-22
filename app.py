import streamlit as st
import pandas as pd
from io import BytesIO
from rapidfuzz import fuzz
from openai import OpenAI

# -------------------------------
# Streamlit Page Setup
# -------------------------------
st.set_page_config(page_title="Jira ↔ TMS Mapper", layout="wide")
st.title("Jira ↔ TMS Mapper")
st.caption("Developed by Pamidi Bala Sumanth")

# -------------------------------
# Step 1: Enter OpenAI API Key
# -------------------------------
api_key = st.text_input("Enter your OpenAI API key", type="password")

if api_key:
    client = OpenAI(api_key=api_key)
else:
    st.warning("⚠️ Please enter your OpenAI API key above to continue.")
    st.stop()

# -------------------------------
# File Uploads
# -------------------------------
jira_file = st.file_uploader("Upload Jira C List (CSV/XLSX)", type=["csv", "xlsx"])
tms_file = st.file_uploader("Upload TMS Export (Excel)", type=["xlsx"])
threshold = st.slider("Select similarity threshold (cosine/ratio)", 0.0, 1.0, 0.60, 0.01)

# -------------------------------
# Helper Functions
# -------------------------------
def get_embeddings(texts, client, model="text-embedding-3-small"):
    embeddings = []
    for txt in texts:
        try:
            response = client.embeddings.create(model=model, input=txt)
            embeddings.append(response.data[0].embedding)
        except Exception as e:
            st.error(f"Embedding error: {e}")
            embeddings.append([0.0] * 1536)
    return embeddings


def cosine_similarity(vec1, vec2):
    import numpy as np
    v1, v2 = np.array(vec1), np.array(vec2)
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10))

# -------------------------------
# Processing Logic
# -------------------------------
if jira_file and tms_file:
    # Load Jira
    if jira_file.name.endswith(".csv"):
        jira = pd.read_csv(jira_file)
    else:
        jira = pd.read_excel(jira_file)

    # Load TMS
    tms = pd.read_excel(tms_file)

    # Normalize text fields
    jira["__bug_text__"] = jira["Summary"].fillna("")
    tms["__case_text__"] = tms["Title"].fillna("") + " " + tms["Description"].fillna("")

    # Generate embeddings
    st.info("⏳ Generating embeddings... this may take some time for large files.")
    jira_embs = get_embeddings(jira["__bug_text__"].tolist(), client)
    tms_embs = get_embeddings(tms["__case_text__"].tolist(), client)

    # Perform matching
    mapped_rows = []
    for i, bug in jira.iterrows():
        bug_emb = jira_embs[i]
        best_score = -1
        best_case = None

        for j, case in tms.iterrows():
            score = cosine_similarity(bug_emb, tms_embs[j])
            if score > best_score:
                best_score = score
                best_case = case

        if best_score >= threshold:
            mapped_rows.append({
                **bug.to_dict(),
                "TMS ID": best_case["ID"],
                "Component": best_case.get("Labels", ""),
                "Platform": "Web",  # Hardcoded fallback
                "TMS Folder Name": best_case.get("Folder", ""),
                "Similarity Score": round(best_score, 4)
            })
        else:
            mapped_rows.append({
                **bug.to_dict(),
                "TMS ID": "❌ No match found – New test case required",
                "Component": "",
                "Platform": "",
                "TMS Folder Name": "",
                "Similarity Score": round(best_score, 4)
            })

    result_df = pd.DataFrame(mapped_rows)

    # -------------------------------
    # Preview in Streamlit
    # -------------------------------
    st.success(f"✅ Complete: {len(result_df)} bugs processed.")
    st.dataframe(result_df, use_container_width=True)

    # -------------------------------
    # Download as Excel
    # -------------------------------
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        result_df.to_excel(writer, index=False, sheet_name="Mapped")
    st.download_button(
        "📥 Download Excel",
        data=output.getvalue(),
        file_name="Jira_C_List_TMS_Final.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
