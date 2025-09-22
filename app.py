import streamlit as st
import pandas as pd
from io import BytesIO
from rapidfuzz import process

# ---------------------------
# App Header
# ---------------------------
st.title("Jira ↔ TMS Mapper")
st.caption("Developed by Pamidi Bala Sumanth")

# ---------------------------
# File Uploads
# ---------------------------
jira_file = st.file_uploader("Upload Jira C List (CSV/XLSX)", type=["csv", "xlsx"])
tms_file = st.file_uploader("Upload TMS Export (Excel)", type=["xlsx"])

# ---------------------------
# Similarity Threshold
# ---------------------------
threshold = st.slider("Select minimum similarity threshold", 0, 100, 60)

if jira_file and tms_file:
    # Load Jira file
    if jira_file.name.endswith(".csv"):
        jira = pd.read_csv(jira_file)
    else:
        jira = pd.read_excel(jira_file)

    # Load TMS file
    tms = pd.read_excel(tms_file)
    id_col = "ID"
    name_col = "Title"
    tms_cases = tms[[id_col, name_col]].dropna()

    # Add new columns
    jira["TMS ID"] = ""
    jira["Platform"] = ""
    jira["Component"] = ""
    jira["Similarity Score"] = 0.0

    # ---------------------------
    # Progress Bar
    # ---------------------------
    progress_text = "🔄 Mapping Jira bugs with TMS cases..."
    progress_bar = st.progress(0, text=progress_text)

    total = len(jira)
    for i, summary in enumerate(jira["Summary"].fillna("")):
        if not summary.strip():
            continue

        # --- Fuzzy matching ---
        match = process.extractOne(
            summary,
            tms_cases[name_col],
            score_cutoff=threshold
        )

        if match:
            best_match, score, idx = match
            jira.at[i, "TMS ID"] = tms_cases.iloc[idx][id_col]
            jira.at[i, "Similarity Score"] = score

        # --- Platform & Component Mapping ---
        text = f"{summary} {jira.at[i,'Description'] if 'Description' in jira.columns else ''}".lower()

        # Platform (only one, first match wins)
        platform_keywords = {
            "mobile": "Mobile",
            "desktop": "Desktop",
            "api": "API",
            "salesforce": "Salesforce",
            "sap": "SAP",
            "web": "Web"
        }
        for kw, plat in platform_keywords.items():
            if kw in text:
                jira.at[i, "Platform"] = plat
                break

        # Components (multiple, comma-separated)
        component_keywords = {
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
            "ai": "AI Research"
        }
        comps = []
        for kw, comp in component_keywords.items():
            if kw in text:
                comps.extend([c.strip() for c in comp.split(",")])
        jira.at[i, "Component"] = ", ".join(sorted(set(comps))) if comps else ""

        # Update progress
        progress_bar.progress(int((i + 1) / total * 100), text=progress_text)

    progress_bar.empty()

    # ---------------------------
    # Final Export
    # ---------------------------
    desired_order = [
        "Issue Type", "Issue key", "Summary", "Assignee", "Reporter",
        "Priority", "Status", "Review Assignee", "Review Comments",
        "TMS ID", "Component", "Platform", "TMS Folder Name",
        "Priority Changed", "Similarity Score"
    ]
    output_df = jira[[col for col in desired_order if col in jira.columns]]

    # Save to Excel in memory
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        output_df.to_excel(writer, index=False, sheet_name="Mapped")
    processed_data = output.getvalue()

    # ---------------------------
    # Preview + Download
    # ---------------------------
    st.success("✅ Mapping complete!")
    mapped_count = output_df["TMS ID"].astype(bool).sum()
    st.info(f"Mapped {mapped_count} out of {len(output_df)} bugs")

    st.subheader("Preview of Mapped Data")
    st.dataframe(output_df.head(50), use_container_width=True)

    st.download_button(
        label="📥 Download Result",
        data=processed_data,
        file_name="Jira_C_List_TMS_Final.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
