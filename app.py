import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from rapidfuzz import process, fuzz

st.set_page_config(page_title="Jira ↔ TMS Mapper", layout="wide")

st.title("Jira ↔ TMS Mapper")
st.caption("Developed by Pamidi Bala Sumanth")

# File uploads
jira_file = st.file_uploader("Upload Jira C List (CSV/XLSX)", type=["csv", "xlsx"])
tms_file = st.file_uploader("Upload TMS Export (Excel)", type=["xlsx"])

similarity_threshold = st.slider(
    "Select similarity threshold (cosine/ratio)",
    0.1, 1.0, 0.6, 0.01
)

if jira_file and tms_file:
    # Load Jira file
    if jira_file.name.endswith(".csv"):
        jira = pd.read_csv(jira_file)
    else:
        jira = pd.read_excel(jira_file)

    # Load TMS export
    tms = pd.read_excel(tms_file, sheet_name=0)

    # Normalize Jira text
    jira["__bug_text__"] = jira["Summary"].fillna("") + " " + jira["Description"].fillna("")

    # Normalize TMS text
    name_col = [c for c in tms.columns if "title" in c.lower() or "name" in c.lower()]
    desc_col = [c for c in tms.columns if "desc" in c.lower()]
    tms["__case_text__"] = (
        tms[name_col[0]].fillna("").astype(str)
        + " " +
        (tms[desc_col[0]].fillna("").astype(str) if desc_col else "")
    )

    # Prepare result storage
    mappings = []

    progress = st.progress(0, text="🔄 Mapping Jira bugs to TMS test cases...")

    # Iterate over Jira bugs
    for idx, row in enumerate(jira.itertuples(index=False)):
        bug_text = str(row.Summary) + " " + str(getattr(row, "Description", ""))
        bug_text = bug_text.strip()

        if not bug_text:
            mappings.append({
                "Issue key": row._asdict().get("Issue key", ""),
                "Summary": row.Summary,
                "TMS ID": "⚠️ No bug text",
                "Similarity Score": 0,
                "Component": "",
                "Platform": "",
                "TMS Folder Name": "Test case needs to be added"
            })
            continue

        # Match with TMS
        best_match = process.extractOne(
            bug_text,
            tms["__case_text__"].tolist(),
            scorer=fuzz.token_set_ratio
        )

        if best_match and best_match[1] / 100 >= similarity_threshold:
            case_idx = tms["__case_text__"].tolist().index(best_match[0])
            case_row = tms.iloc[case_idx]

            mappings.append({
                "Issue key": row._asdict().get("Issue key", ""),
                "Summary": row.Summary,
                "TMS ID": case_row.get("ID", ""),
                "Similarity Score": round(best_match[1] / 100, 4),
                "Component": row._asdict().get("Component", ""),
                "Platform": row._asdict().get("Platform", ""),
                "TMS Folder Name": case_row.get("Folder", "")
            })
        else:
            mappings.append({
                "Issue key": row._asdict().get("Issue key", ""),
                "Summary": row.Summary,
                "TMS ID": "❌ No match found",
                "Similarity Score": 0,
                "Component": row._asdict().get("Component", ""),
                "Platform": row._asdict().get("Platform", ""),
                "TMS Folder Name": "Test case needs to be added"
            })

        progress.progress((idx + 1) / len(jira),
                          text=f"Processed {idx+1}/{len(jira)} Jira bugs")

    progress.empty()

    # Final DataFrame
    mapped = pd.DataFrame(mappings)

    # Status
    matched_count = (mapped["TMS ID"] != "❌ No match found").sum()
    new_needed = (mapped["TMS ID"] == "❌ No match found").sum()

    st.success(f"✅ Complete: {matched_count}/{len(mapped)} mapped to existing TMS. {new_needed} need new test cases.")

    # Show ALL rows in preview
    st.write(mapped.to_html(escape=False, index=False), unsafe_allow_html=True)

    # ✅ Export to Excel in memory
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        mapped.to_excel(writer, index=False, sheet_name="Mapped")
    processed_data = output.getvalue()

    st.download_button(
        label="⬇️ Download Excel",
        data=processed_data,
        file_name="Jira_C_List_TMS_Final.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
