import streamlit as st
import pandas as pd
from io import BytesIO
from rapidfuzz import fuzz, process

st.set_page_config(page_title="Jira ↔ TMS Mapper", page_icon="📝")

# --- Author banner ---
st.markdown("""
# Jira ↔ TMS Mapper  
**Developed by Pamidi Bala Sumanth**
""")

# --- File uploaders ---
jira_file = st.file_uploader("Upload Jira C List (CSV/XLSX)", type=["csv", "xlsx"])
tms_file = st.file_uploader("Upload TMS Export (Excel)", type=["xlsx"])

# --- Keyword-based rules ---
PLATFORM_RULES = {
    "browser": "Web", "chrome": "Web", "edge": "Web", "firefox": "Web",
    "android": "Mobile", "ios": "Mobile", "mobile": "Mobile",
    "desktop": "Desktop", "windows": "Desktop", "mac": "Desktop",
    "api": "API", "endpoint": "API", "rest": "API",
    "salesforce": "Salesforce",
    "sap": "SAP",
    "accessibility": "Accessibility",
    "visual": "Visual",
}

COMPONENT_RULES = {
    "addons": "Add-Ons",
    "user": "User Management", "auth": "User Management", "login": "User Management",
    "tunnel": "Tunnel",
    "recorder": "Recorder (Browser)",
    "api": "Public APIs",
    "nlp": "NLP",
    "execution": "Execution",
    "editor": "Live Editor",
    "integration": "Integrations",
    "billing": "Billing & Licenses", "license": "Billing & Licenses",
    "heal": "Auto Heal",
    "autonomous": "Autonomous",
    "copilot": "Co-pilot",
    "documentation": "Documentation",
    "sap": "SAP",
    "salesforce": "SalesForce",
    "project": "Projects",
    "agent": "Terminal & Agent",
    "plan": "Test Plans and Runs",
    "author": "Test Authoring",
    "atto": "Atto Generator, Atto Live Editor, Atto Analyzer",
    "ai": "AI Research",
}

# --- Helper functions ---
def assign_platform(text: str) -> str:
    """Return a single platform (first match wins)"""
    text = text.lower()
    for k, v in PLATFORM_RULES.items():
        if k in text:
            return v
    return "Internal"

def assign_components(text: str) -> str:
    """Return multiple comma-separated components"""
    text = text.lower()
    comps = set()
    for k, v in COMPONENT_RULES.items():
        if k in text:
            for part in v.split(","):
                comps.add(part.strip())
    return ", ".join(sorted(comps)) if comps else ""

def map_tms_id(jira_text, tms_df):
    """Return a single best TMS ID"""
    choices = tms_df["Title"].fillna("").tolist()
    best_match, score, idx = process.extractOne(jira_text, choices, scorer=fuzz.partial_ratio)
    if score > 60:  # threshold for match
        return tms_df.iloc[idx]["ID"]
    return ""

# --- Main workflow ---
if jira_file and tms_file:
    # Load Jira
    if jira_file.name.endswith(".csv"):
        jira = pd.read_csv(jira_file)
    else:
        jira = pd.read_excel(jira_file)

    # Load TMS
    tms = pd.read_excel(tms_file, sheet_name="Test Cases")

    # Create working text for matching
    jira["__text__"] = jira["Summary"].fillna("") + " " + jira["Description"].fillna("")

    # Map TMS IDs
    jira["TMS ID"] = jira["__text__"].apply(lambda x: map_tms_id(x, tms))

    # Assign Platform (only one)
    jira["Platform"] = jira["__text__"].apply(assign_platform)

    # Assign Components (can be many, comma-separated)
    jira["Component"] = jira["__text__"].apply(assign_components)

    # Drop helper
    jira.drop(columns=["__text__"], inplace=True)

    # ✅ Reorder columns for final output
    desired_order = [
        "Issue Type", "Issue key", "Summary", "Assignee", "Reporter",
        "Priority", "Status", "Review Assignee", "Review Comments",
        "TMS ID", "Component", "Platform", "TMS Folder Name", "Priority Changed"
    ]
    jira = jira.reindex(columns=desired_order)

    # ✅ Show preview in UI
    st.subheader("Preview of Mapped Data")
    st.dataframe(jira.head(20))

    # ✅ Export
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
