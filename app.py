import streamlit as st
import pandas as pd
from io import BytesIO

st.title("Jira ↔ TMS Mapper")

jira_file = st.file_uploader("Upload Jira C List (Excel/CSV)", type=["csv", "xlsx"])
tms_file = st.file_uploader("Upload TMS Export (Excel)", type=["xlsx"])

if jira_file and tms_file:
    # Load Jira file
    if jira_file.name.endswith(".csv"):
        jira = pd.read_csv(jira_file)
    else:
        jira = pd.read_excel(jira_file)

    # Load TMS export
    tms = pd.read_excel(tms_file, sheet_name="Test Cases")

    # Placeholder mapping logic
    jira["TMS ID"] = ""

    # ✅ Export to Excel in memory
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        jira.to_excel(writer, index=False, sheet_name="Mapped")
    processed_data = output.getvalue()

    st.success("Mapping complete!")
    st.download_button(
        label="Download Result",
        data=processed_data,
        file_name="Jira_C_List_TMS_Final.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
