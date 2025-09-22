import streamlit as st
import pandas as pd

st.title("Jira ↔ TMS Mapper")

jira_file = st.file_uploader("Upload Jira C List (Excel/CSV)", type=["csv", "xlsx"])
tms_file = st.file_uploader("Upload TMS Export (Excel)", type=["xlsx"])

if jira_file and tms_file:
    if jira_file.name.endswith(".csv"):
        jira = pd.read_csv(jira_file)
    else:
        jira = pd.read_excel(jira_file)

    tms = pd.read_excel(tms_file, sheet_name="Test Cases")

    # TODO: add real mapping logic
    jira["TMS ID"] = ""

    st.success("Mapping complete!")
    st.download_button(
        "Download Result",
        jira.to_excel(index=False, engine="xlsxwriter"),
        file_name="Jira_C_List_TMS_Final.xlsx"
    )
