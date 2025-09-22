import streamlit as st
import pandas as pd
from rapidfuzz import fuzz, process
from io import BytesIO

st.title("Jira ↔ TMS Mapper")
st.markdown("**Developed by Pamidi Bala Sumanth**")

# Upload files
jira_file = st.file_uploader("Upload Jira C List (CSV/XLSX)", type=["csv", "xlsx"])
tms_file = st.file_uploader("Upload TMS Export (Excel)", type=["xlsx"])

# Similarity threshold slider
threshold = st.slider(
    "Select minimum similarity threshold",
    min_value=40,
    max_value=95,
    value=60,
    step=5,
    help="Only matches above this score will be mapped. Lower = more matches, Higher = stricter."
)

if jira_file and tms_file:
    # Load Jira
    if jira_file.name.endswith(".csv"):
        jira = pd.read_csv(jira_file)
    else:
        jira = pd.read_excel(jira_file)

    # Load TMS (first sheet = Test Cases usually)
    tms = pd.read_excel(tms_file, sheet_name=0)

    # Debug: show TMS columns in the UI
    st.write("🔎 Detected TMS sheet columns:", list(tms.columns))

    # Try to auto-detect ID and Name columns
    id_col = [c for c in tms.columns if "id" in c.lower()][0]
    name_col = [c for c in tms.columns if "name" in c.lower()][0]

    tms_cases = tms[[id_col, name_col]].dropna()

    # Add new columns to Jira
    jira["TMS ID"] = ""
    jira["Similarity Score"] = 0
    jira["Components"] = ""
    jira["Platform"] = ""
    jira["TMS Folder Name"] = ""
    jira["Priority Changed"] = ""

    # Progress bar
    progress = st.progress(0)
    total = len(jira)

    for i, row in jira.iterrows():
        bug_summary = str(row.get("Summary", ""))

        # Fuzzy match
        best_match = process.extractOne(
            bug_summary,
            tms_cases[name_col],
            scorer=fuzz.token_sort_ratio
        )

        if best_match:
            match_name, score, idx = best_match
            if score >= threshold:
                match_id = tms_cases.iloc[idx][id_col]
                jira.at[i, "TMS ID"] = match_id
                jira.at[i, "Similarity Score"] = score

                # Component mapping rules
                comps = []
                if "Salesforce" in bug_summary: comps.append("SalesForce")
                if "SAP" in bug_summary: comps.append("SAP")
                if "API" in bug_summary: comps.append("Public APIs")
                if "Recorder" in bug_summary: comps.append("Recorder (Browser)")
                jira.at[i, "Components"] = ", ".join(comps) if comps else "Unassigned"

                # Platform detection rules
                if "mobile" in bug_summary.lower():
                    jira.at[i, "Platform"] = "Mobile"
                elif "desktop" in bug_summary.lower():
                    jira.at[i, "Platform"] = "Desktop"
                elif "api" in bug_summary.lower():
                    jira.at[i, "Platform"] = "API"
                elif "salesforce" in bug_summary.lower():
                    jira.at[i, "Platform"] = "Salesforce"
                elif "sap" in bug_summary.lower():
                    jira.at[i, "Platform"] = "SAP"
                else:
                    jira.at[i, "Platform"] = "Web"

        # Update progress
        progress.progress((i + 1) / total)

    progress.empty()

    # Reorder final columns
    desired_order = [
        "Issue Type", "Issue key", "Summary", "Assignee", "Reporter",
        "Priority", "Status", "Review Assignee", "Review Comments",
        "TMS ID", "Components", "Platform", "TMS Folder Name",
        "Priority Changed", "Similarity Score"
    ]
    jira = jira.reindex(columns=desired_order)

    # Show preview
    st.subheader("Preview of Mapped Data")
    st.dataframe(jira.head(20))

    # Summary count
    mapped_count = jira["TMS ID"].astype(bool).sum()
    st.info(f"✅ Successfully mapped {mapped_count} out of {len(jira)} bugs.")

    # Export to Excel
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        jira.to_excel(writer, index=False, sheet_name="Mapped")
    processed_data = output.getvalue()

    st.success("✅ Mapping complete!")
    st.download_button(
        label="Download Result",
        data=processed_data,
        file_name="Jira_C_List_TMS_Final.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
