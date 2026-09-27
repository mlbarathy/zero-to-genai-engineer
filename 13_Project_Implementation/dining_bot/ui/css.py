"""Streamlit light-theme CSS."""

import streamlit as st

def inject_app_css() -> None:
    """Light theme with explicit text colors — avoids dark-mode white-on-white bugs."""
    st.markdown(
        """
<style>
  :root { color-scheme: light; }
  .stApp, [data-testid="stAppViewContainer"], .main {
    background-color: #f4f4f5;
    color: #18181b;
  }
  .main .block-container {
    max-width: 42rem;
    padding-top: 0.75rem;
    padding-bottom: 5rem;
    color: #18181b;
  }
  .main p, .main h1, .main h2, .main h3, .main h4,
  .main li, .main span, .main label, .main code {
    color: #18181b;
  }
  [data-testid="stCaptionContainer"] p,
  [data-testid="stMarkdownContainer"] p,
  [data-testid="stMarkdownContainer"] h3 {
    color: #18181b !important;
  }

  [data-testid="stSidebar"] {
    background-color: #ffffff;
    border-right: 1px solid #e4e4e7;
    color: #18181b;
  }
  [data-testid="stSidebar"] .block-container { max-width: 100%; padding-top: 1rem; }
  [data-testid="stSidebar"] p,
  [data-testid="stSidebar"] span,
  [data-testid="stSidebar"] label,
  [data-testid="stSidebar"] h1,
  [data-testid="stSidebar"] h2,
  [data-testid="stSidebar"] h3,
  [data-testid="stSidebar"] code {
    color: #18181b !important;
  }
  [data-testid="stSidebar"] .stTextInput input {
    color: #18181b !important;
    background: #ffffff !important;
    border: 1px solid #d4d4d8 !important;
  }

  .stButton > button {
    color: #18181b !important;
    background-color: #ffffff !important;
    border: 1px solid #d4d4d8 !important;
  }
  .stButton > button:hover {
    color: #18181b !important;
    background-color: #f4f4f5 !important;
    border-color: #a1a1aa !important;
  }
  .stButton > button p,
  .stButton > button span,
  .stButton > button div {
    color: #18181b !important;
  }
  button[kind="primary"] {
    color: #ffffff !important;
    background-color: #2563eb !important;
  }
  button[kind="primary"] p { color: #ffffff !important; }

  /* Plan file link — popover trigger styled as a hyperlink */
  [data-testid="stPopover"] > button {
    color: #2563eb !important;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
    min-height: 0 !important;
    font-weight: 500 !important;
    text-decoration: underline !important;
    text-underline-offset: 2px !important;
    width: auto !important;
  }
  [data-testid="stPopover"] > button:hover {
    color: #1d4ed8 !important;
    background: transparent !important;
  }
  [data-testid="stPopover"] > button p,
  [data-testid="stPopover"] > button span,
  [data-testid="stPopover"] > button div {
    color: #2563eb !important;
  }
  /* Popover panel — fixed size so opening it does not reflow the chat column */
  [data-testid="stPopoverBody"] {
    width: min(36rem, 92vw) !important;
    max-height: min(70vh, 32rem) !important;
    overflow-y: auto !important;
    overflow-x: hidden !important;
    word-break: break-word !important;
  }
  [data-testid="stPopoverBody"] p,
  [data-testid="stPopoverBody"] li,
  [data-testid="stPopoverBody"] h1,
  [data-testid="stPopoverBody"] h2,
  [data-testid="stPopoverBody"] h3 {
    color: #18181b !important;
  }

  footer, #MainMenu { visibility: hidden; height: 0; }
  [data-testid="stHeader"] { background: transparent; }
  [data-testid="stToolbar"] { display: none; }

  [data-testid="stChatMessage"] {
    background: transparent !important;
    padding: 0.2rem 0 !important;
  }
  [data-testid="stChatMessageContent"] {
    background: #ffffff !important;
    color: #18181b !important;
    border: 1px solid #e4e4e7 !important;
    border-radius: 1rem !important;
    padding: 0.75rem 1rem !important;
  }
  [data-testid="stChatMessageContent"] p,
  [data-testid="stChatMessageContent"] li,
  [data-testid="stChatMessageContent"] span,
  [data-testid="stChatMessageContent"] strong {
    color: #18181b !important;
  }
  [data-testid="stChatInput"] textarea {
    color: #18181b !important;
    -webkit-text-fill-color: #18181b !important;
    background: #ffffff !important;
    caret-color: #18181b !important;
  }
  [data-testid="stChatInput"] > div {
    border-color: #d4d4d8 !important;
    border-radius: 1.25rem !important;
    background: #ffffff !important;
  }
  [data-testid="stStatus"] {
    border: 1px solid #e4e4e7;
    border-radius: 0.75rem;
    background: #fafafa !important;
    color: #18181b !important;
  }
  [data-testid="stStatus"] p,
  [data-testid="stStatus"] span,
  [data-testid="stStatus"] label {
    color: #18181b !important;
  }
  [data-testid="stExpander"] summary,
  [data-testid="stExpander"] p {
    color: #18181b !important;
  }

  .db-plan-shell {
    font-family: ui-sans-serif, system-ui, sans-serif;
    border: 1px solid #e4e4e7;
    border-radius: 0.75rem;
    background: #fafafa;
    padding: 12px 14px;
    margin: 0.5rem 0 0.75rem;
    color: #18181b;
  }
  .db-plan-head {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #52525b;
    margin-bottom: 10px;
  }
  .db-plan-steps { position: relative; }
  .db-plan-step { display: flex; gap: 10px; padding: 6px 0; position: relative; }
  .db-plan-step.nested { margin-left: 20px; padding-left: 10px; border-left: 2px solid #e4e4e7; }
  .db-plan-rail { width: 22px; flex-shrink: 0; display: flex; justify-content: center; }
  .db-plan-badge {
    width: 20px; height: 20px; border-radius: 999px;
    display: flex; align-items: center; justify-content: center;
    font-size: 10px; font-weight: 700; border: 1px solid transparent;
  }
  .db-plan-step.running .db-plan-badge {
    background: #eff6ff; border-color: #93c5fd; color: #2563eb;
    animation: db-pulse 1.2s ease-in-out infinite;
  }
  .db-plan-step.done .db-plan-badge { background: #ecfdf5; border-color: #86efac; color: #16a34a; }
  .db-plan-step.error .db-plan-badge { background: #fef2f2; border-color: #fca5a5; color: #dc2626; }
  .db-plan-title { font-size: 0.84rem; font-weight: 600; color: #18181b; line-height: 1.35; }
  .db-plan-step.running .db-plan-title { color: #2563eb; }
  .db-plan-detail { font-size: 0.78rem; color: #52525b; margin-top: 2px; line-height: 1.45; word-break: break-word; }
  .db-plan-meta { font-size: 0.7rem; color: #71717a; margin-top: 2px; }
  .db-plan-chip { font-size: 0.7rem; color: #52525b; background: #e4e4e7; border-radius: 999px; padding: 1px 6px; }
  .db-plan-title-row { display: flex; align-items: baseline; gap: 6px; flex-wrap: wrap; }
  @keyframes db-pulse { 0%,100%{opacity:1} 50%{opacity:.5} }
  @keyframes db-spin { to { transform: rotate(360deg); } }
  .db-plan-step.running .db-plan-badge.spin::after {
    content: ""; width: 10px; height: 10px;
    border: 2px solid #93c5fd; border-top-color: #2563eb;
    border-radius: 50%; animation: db-spin 0.8s linear infinite;
  }
</style>
        """,
        unsafe_allow_html=True,
    )


def inject_planning_progress_css() -> None:
    """Backward-compatible alias."""
    inject_app_css()
