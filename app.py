import hashlib

import streamlit as st

from foundry_local_sdk import Configuration, FoundryLocalManager

from src.chat import get_chat_model, answer_query
from src.embeddings import get_embedding_model
from src.ingest import ingest_pdf_bytes


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Sidera",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# SESSION STATE
# =========================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "theme_selector" not in st.session_state:
    st.session_state.theme_selector = "Dark"

if "indexed_upload_hash" not in st.session_state:
    st.session_state.indexed_upload_hash = None

if "active_document" not in st.session_state:
    st.session_state.active_document = None

if "active_document_chunks" not in st.session_state:
    st.session_state.active_document_chunks = 0


# =========================================================
# THEME CALLBACK
# =========================================================

def update_theme():
    st.session_state.theme = (
        st.session_state.theme_selector.lower()
    )


if "theme" not in st.session_state:
    st.session_state.theme = "dark"


# =========================================================
# THEME SELECTOR
# =========================================================

st.sidebar.radio(
    "Appearance",
    ["Dark", "Light"],
    horizontal=True,
    key="theme_selector",
    on_change=update_theme,
)

theme = st.session_state.theme


# =========================================================
# THEME VARIABLES
# =========================================================

if theme == "dark":

    BG = "#050816"
    BG_SECONDARY = "#080d1d"
    SIDEBAR = "#060a16"

    CARD = "rgba(12, 18, 38, 0.90)"

    TEXT = "#f8fafc"
    TEXT_SECONDARY = "#a5b4cf"
    TEXT_MUTED = "#71809d"

    BORDER = "rgba(139, 92, 246, 0.22)"

    ACCENT = "#8b5cf6"

    INPUT = "#0b1022"

    HERO_START = "#080d20"
    HERO_END = "#10143a"

    SHADOW = "rgba(0, 0, 0, 0.28)"

    MODEL_BG = "#0b1022"
    MODEL_TEXT = "#c4b5fd"

else:

    BG = "#f7f8fc"
    BG_SECONDARY = "#eef2ff"
    SIDEBAR = "#ffffff"

    CARD = "rgba(255, 255, 255, 0.94)"

    TEXT = "#111827"
    TEXT_SECONDARY = "#53627a"
    TEXT_MUTED = "#8491a7"

    BORDER = "rgba(99, 102, 241, 0.16)"

    ACCENT = "#7c3aed"

    INPUT = "#f1f5f9"

    HERO_START = "#ffffff"
    HERO_END = "#eef2ff"

    SHADOW = "rgba(15, 23, 42, 0.08)"

    MODEL_BG = "#f1f5f9"
    MODEL_TEXT = "#6d28d9"


# =========================================================
# CSS
# =========================================================

st.markdown(
    f"""
<style>

/* GLOBAL */

.stApp {{
    background:
        radial-gradient(
            circle at 80% 5%,
            rgba(124,58,237,0.14),
            transparent 28%
        ),
        radial-gradient(
            circle at 15% 90%,
            rgba(99,102,241,0.08),
            transparent 30%
        ),
        {BG};

    color: {TEXT};

    transition:
        background 0.3s ease,
        color 0.3s ease;
}}

.block-container {{
    max-width: 1180px;
    padding-top: 2.2rem;
    padding-bottom: 7rem;
}}


/* SIDEBAR */

section[data-testid="stSidebar"] {{
    background:
        linear-gradient(
            180deg,
            {SIDEBAR},
            {BG_SECONDARY}
        );

    border-right:
        1px solid {BORDER};
}}

section[data-testid="stSidebar"] * {{
    color: {TEXT};
}}


/* RADIO */

div[role="radiogroup"] {{
    background:
        rgba(124,58,237,0.06);

    border:
        1px solid {BORDER};

    border-radius: 12px;

    padding: 5px 8px;

    margin-bottom: 12px;
}}


/* BRAND */

.sidera-brand {{
    font-size: 1.7rem;
    font-weight: 850;
    letter-spacing: 0.22em;
    color: {TEXT};
    margin-bottom: 4px;
}}

.sidera-brand-star {{
    display: inline-block;
    color: {ACCENT};
    margin-right: 8px;

    animation:
        starPulse 2.5s ease-in-out infinite;
}}

.sidera-brand-subtitle {{
    color: {TEXT_SECONDARY};
    font-size: 0.82rem;
    margin-bottom: 20px;
}}


/* STATUS */

.status-card {{
    background:
        rgba(16,185,129,0.07);

    border:
        1px solid rgba(52,211,153,0.28);

    border-radius: 14px;

    padding: 14px 15px;

    margin-bottom: 18px;
}}

.status-main {{
    color: #34d399;
    font-size: 0.85rem;
    font-weight: 700;
    margin-bottom: 3px;
}}

.status-sub {{
    color: {TEXT_MUTED};
    font-size: 0.72rem;
}}


/* CUSTOM MODEL BOX */

.model-box {{
    background: {MODEL_BG};

    border:
        1px solid {BORDER};

    border-radius: 10px;

    padding: 9px 11px;

    margin-top: 6px;
    margin-bottom: 16px;

    color: {MODEL_TEXT};

    font-family:
        ui-monospace,
        SFMono-Regular,
        Menlo,
        Monaco,
        Consolas,
        monospace;

    font-size: 0.78rem;
}}


/* HERO */

.hero {{
    position: relative;
    overflow: hidden;

    background:
        radial-gradient(
            circle at 86% 28%,
            rgba(124,58,237,0.22),
            transparent 26%
        ),
        radial-gradient(
            circle at 73% 60%,
            rgba(99,102,241,0.13),
            transparent 24%
        ),
        linear-gradient(
            135deg,
            {HERO_START},
            {HERO_END}
        );

    border:
        1px solid {BORDER};

    border-radius: 26px;

    padding: 48px;

    box-shadow:
        0 24px 65px {SHADOW};

    margin-bottom: 26px;

    animation:
        fadeUp 0.65s ease;
}}

.hero::before {{
    content:
        "✦   ·    ✧      ·     ✦        ·    ✧        ·     ✦";

    position: absolute;

    right: 4%;
    top: 14%;

    width: 38%;

    color: {ACCENT};

    opacity: 0.30;

    font-size: 13px;

    letter-spacing: 11px;
    line-height: 3;

    animation:
        floatStars 7s ease-in-out infinite;
}}

.hero::after {{
    content: "";

    position: absolute;

    width: 350px;
    height: 350px;

    right: -85px;
    bottom: -210px;

    border-radius: 50%;

    background:
        radial-gradient(
            circle,
            rgba(139,92,246,0.28),
            rgba(99,102,241,0.09) 42%,
            transparent 68%
        );

    animation:
        planetGlow 5s ease-in-out infinite;
}}

.hero-stars {{
    color: {ACCENT};
    font-size: 0.86rem;
    letter-spacing: 0.55em;
    margin-bottom: 18px;
}}

.hero-title {{
    color: {TEXT};

    font-size: 3.5rem;

    font-weight: 850;

    line-height: 1;

    letter-spacing: -0.055em;

    margin-bottom: 16px;

    position: relative;

    z-index: 2;
}}

.hero-tagline {{
    color: {ACCENT};

    font-size: 1.1rem;

    font-weight: 650;

    margin-bottom: 12px;

    position: relative;

    z-index: 2;
}}

.hero-description {{
    max-width: 720px;

    color: {TEXT_SECONDARY};

    font-size: 0.94rem;

    line-height: 1.75;

    position: relative;

    z-index: 2;
}}


/* BADGES */

.badges {{
    margin-top: 25px;
    position: relative;
    z-index: 2;
}}

.badge {{
    display: inline-block;

    padding: 7px 12px;

    margin-right: 7px;
    margin-bottom: 5px;

    border-radius: 999px;

    font-size: 0.68rem;

    font-weight: 750;

    letter-spacing: 0.055em;
}}

.badge-local {{
    color: #34d399;

    background:
        rgba(16,185,129,0.08);

    border:
        1px solid rgba(52,211,153,0.40);
}}

.badge-rag {{
    color: {ACCENT};

    background:
        rgba(124,58,237,0.08);

    border:
        1px solid rgba(139,92,246,0.35);
}}

.badge-private {{
    color: {TEXT_SECONDARY};

    background:
        rgba(100,116,139,0.07);

    border:
        1px solid {BORDER};
}}


/* FEATURE CARDS */

.feature-card {{
    background: {CARD};

    backdrop-filter:
        blur(14px);

    border:
        1px solid {BORDER};

    border-radius: 19px;

    padding: 24px;

    min-height: 165px;

    box-shadow:
        0 10px 30px {SHADOW};

    transition:
        transform 0.25s ease,
        border-color 0.25s ease,
        box-shadow 0.25s ease;

    animation:
        fadeUp 0.8s ease;
}}

.feature-card:hover {{
    transform:
        translateY(-5px);

    border-color:
        {ACCENT};

    box-shadow:
        0 18px 45px
        rgba(99,102,241,0.13);
}}

.feature-symbol {{
    width: 40px;
    height: 40px;

    display: flex;

    align-items: center;
    justify-content: center;

    border-radius: 12px;

    background:
        rgba(124,58,237,0.12);

    color: {ACCENT};

    font-size: 1.15rem;

    margin-bottom: 17px;

    transition:
        transform 0.25s ease;
}}

.feature-card:hover .feature-symbol {{
    transform:
        rotate(8deg)
        scale(1.08);
}}

.feature-title {{
    color: {TEXT};

    font-size: 0.96rem;

    font-weight: 750;

    margin-bottom: 8px;
}}

.feature-text {{
    color: {TEXT_SECONDARY};

    font-size: 0.84rem;

    line-height: 1.65;
}}


/* WELCOME */

.welcome-card {{
    margin-top: 28px;
    margin-bottom: 18px;

    padding: 28px;

    text-align: center;

    background:
        linear-gradient(
            135deg,
            rgba(124,58,237,0.06),
            rgba(99,102,241,0.03)
        );

    border:
        1px solid {BORDER};

    border-radius: 20px;

    animation:
        fadeUp 1s ease;
}}

.welcome-star {{
    color: {ACCENT};

    font-size: 1.25rem;

    animation:
        starPulse 2.3s
        ease-in-out infinite;
}}

.welcome-title {{
    color: {ACCENT};

    font-size: 1.08rem;

    font-weight: 700;

    margin-top: 8px;
}}

.welcome-text {{
    color: {TEXT_SECONDARY};

    font-size: 0.84rem;

    margin-top: 5px;
}}


/* CHAT */

div[data-testid="stChatMessage"] {{
    background: {CARD};

    backdrop-filter:
        blur(14px);

    border:
        1px solid {BORDER};

    border-radius: 18px;

    padding: 16px 19px;

    margin-bottom: 12px;

    animation:
        messageAppear 0.35s ease;

    box-shadow:
        0 5px 18px {SHADOW};
}}

div[data-testid="stChatInput"] {{
    background: transparent;
}}

div[data-testid="stChatInput"] > div {{
    background: {INPUT};

    border:
        1px solid {BORDER};

    border-radius: 17px;
}}

div[data-testid="stChatInput"] > div:focus-within {{
    border-color: {ACCENT};

    box-shadow:
        0 0 0 3px
        rgba(124,58,237,0.10),
        0 0 28px
        rgba(124,58,237,0.10);
}}


/* BUTTON */

.stButton > button {{
    width: 100%;

    border-radius: 11px;

    border:
        1px solid {BORDER};

    background:
        {CARD};

    color: {TEXT};

    transition:
        all 0.22s ease;
}}

.stButton > button:hover {{
    border-color: {ACCENT};
    color: {ACCENT};

    transform:
        translateY(-1px);
}}


/* EXPANDERS */

details {{
    background:
        {CARD} !important;

    border:
        1px solid {BORDER} !important;

    border-radius:
        13px !important;
}}


/* FOOTER */

.sidera-footer {{
    text-align: center;

    color: {TEXT_MUTED};

    font-size: 0.72rem;

    margin-top: 38px;

    padding-top: 20px;

    border-top:
        1px solid {BORDER};
}}


/* ANIMATIONS */

@keyframes fadeUp {{
    from {{
        opacity: 0;
        transform: translateY(12px);
    }}

    to {{
        opacity: 1;
        transform: translateY(0);
    }}
}}

@keyframes messageAppear {{
    from {{
        opacity: 0;
        transform: translateY(7px);
    }}

    to {{
        opacity: 1;
        transform: translateY(0);
    }}
}}

@keyframes starPulse {{
    0%, 100% {{
        opacity: 0.65;
        transform: scale(1);
    }}

    50% {{
        opacity: 1;

        transform:
            scale(1.16);

        text-shadow:
            0 0 15px {ACCENT};
    }}
}}

@keyframes floatStars {{
    0%, 100% {{
        transform:
            translateY(0);

        opacity: 0.24;
    }}

    50% {{
        transform:
            translateY(-8px);

        opacity: 0.55;
    }}
}}

@keyframes planetGlow {{
    0%, 100% {{
        opacity: 0.55;
        transform: scale(1);
    }}

    50% {{
        opacity: 0.9;
        transform: scale(1.08);
    }}
}}






/* =========================================================
   UPLOAD FIRST EXPERIENCE
   ========================================================= */

.upload-gate {{
    min-height: 58vh;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 2.2rem 1rem 3rem;
}}

.upload-gate-card {{
    width: min(760px, 96%);
    position: relative;
    overflow: hidden;
    padding: 42px 42px 34px;
    text-align: center;
    border-radius: 26px;
    background:
        radial-gradient(
            circle at 82% 18%,
            rgba(124,58,237,0.20),
            transparent 30%
        ),
        linear-gradient(
            145deg,
            {HERO_START},
            {HERO_END}
        );
    border: 1px solid {BORDER};
    box-shadow: 0 24px 65px {SHADOW};
}}

.upload-gate-card::after {{
    content: "";
    position: absolute;
    width: 300px;
    height: 300px;
    right: -150px;
    bottom: -180px;
    border-radius: 50%;
    background:
        radial-gradient(
            circle,
            rgba(139,92,246,0.28),
            rgba(99,102,241,0.08) 45%,
            transparent 72%
        );
}}

.upload-gate-icon {{
    position: relative;
    z-index: 2;
    width: 64px;
    height: 64px;
    display: flex;
    align-items: center;
    justify-content: center;
    margin: 0 auto 18px;
    border-radius: 18px;
    color: {ACCENT};
    background: rgba(124,58,237,0.11);
    border: 1px solid rgba(139,92,246,0.25);
    font-size: 1.65rem;
    animation: starPulse 2.2s ease-in-out infinite;
}}

.upload-gate-title {{
    position: relative;
    z-index: 2;
    color: {TEXT};
    font-size: 2rem;
    font-weight: 820;
    letter-spacing: -0.04em;
    margin-bottom: 10px;
}}

.upload-gate-subtitle {{
    position: relative;
    z-index: 2;
    max-width: 590px;
    margin: 0 auto;
    color: {TEXT_SECONDARY};
    font-size: 0.92rem;
    line-height: 1.7;
}}

.upload-gate-steps {{
    position: relative;
    z-index: 2;
    display: flex;
    justify-content: center;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 24px;
}}

.upload-step {{
    padding: 7px 11px;
    border-radius: 999px;
    color: {TEXT_SECONDARY};
    background: rgba(124,58,237,0.055);
    border: 1px solid {BORDER};
    font-size: 0.69rem;
    font-weight: 650;
}}

.upload-gate-note {{
    position: relative;
    z-index: 2;
    margin-top: 18px;
    color: {TEXT_MUTED};
    font-size: 0.72rem;
}}

.upload-active-card {{
    margin: 8px 0 22px;
    padding: 14px 16px;
    border-radius: 14px;
    background: rgba(16,185,129,0.065);
    border: 1px solid rgba(52,211,153,0.25);
    color: {TEXT_SECONDARY};
    font-size: 0.80rem;
}}

.upload-active-card strong {{
    color: #34d399;
}}


/* =========================================================
   DOCUMENT UPLOAD
   ========================================================= */

.upload-shell {{
    margin-top: 4px;
    margin-bottom: 24px;
    padding: 20px 22px 12px;
    border-radius: 18px;
    background: {CARD};
    border: 1px solid {BORDER};
    box-shadow: 0 10px 30px {SHADOW};
}}

.upload-heading {{
    color: {TEXT};
    font-size: 0.96rem;
    font-weight: 750;
    margin-bottom: 4px;
}}

.upload-subtitle {{
    color: {TEXT_SECONDARY};
    font-size: 0.80rem;
    line-height: 1.55;
    margin-bottom: 10px;
}}

[data-testid="stFileUploader"] {{
    margin-top: -6px;
}}

[data-testid="stFileUploaderDropzone"] {{
    background: {INPUT} !important;
    border: 1px dashed {BORDER} !important;
    border-radius: 14px !important;
}}

[data-testid="stFileUploaderDropzone"]:hover {{
    border-color: {ACCENT} !important;
}}

[data-testid="stFileUploaderDropzone"] *,
[data-testid="stFileUploaderFile"] * {{
    color: {TEXT} !important;
}}

[data-testid="stFileUploaderDropzone"] small {{
    color: {TEXT_MUTED} !important;
}}

[data-testid="stFileUploaderDropzone"] button {{
    background: rgba(124,58,237,0.10) !important;
    color: {TEXT} !important;
    border: 1px solid {BORDER} !important;
}}

.upload-ready {{
    margin-top: 8px;
    margin-bottom: 12px;
    padding: 11px 13px;
    border-radius: 12px;
    background: rgba(16,185,129,0.07);
    border: 1px solid rgba(52,211,153,0.25);
    color: {TEXT_SECONDARY};
    font-size: 0.78rem;
}}

.upload-ready strong {{
    color: #34d399;
}}


/* =========================================================
   SIDERA STARTUP LOADER
   ========================================================= */

.startup-wrap {{
    min-height: 58vh;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 3rem 1rem 4rem;
}}

.startup-card {{
    width: min(620px, 92%);
    position: relative;
    overflow: hidden;
    text-align: center;
    padding: 42px 36px 36px;
    border-radius: 24px;
    background:
        radial-gradient(
            circle at 82% 20%,
            rgba(124,58,237,0.20),
            transparent 28%
        ),
        linear-gradient(
            145deg,
            {HERO_START},
            {HERO_END}
        );
    border: 1px solid {BORDER};
    box-shadow: 0 24px 65px {SHADOW};
    animation: startupFade 0.45s ease;
}}

.startup-card::after {{
    content: "";
    position: absolute;
    width: 240px;
    height: 240px;
    right: -110px;
    bottom: -145px;
    border-radius: 50%;
    background:
        radial-gradient(
            circle,
            rgba(139,92,246,0.30),
            rgba(99,102,241,0.08) 45%,
            transparent 70%
        );
    animation: startupGlow 3.2s ease-in-out infinite;
}}

.startup-star {{
    position: relative;
    z-index: 2;
    width: 58px;
    height: 58px;
    margin: 0 auto 18px;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 18px;
    color: {ACCENT};
    background: rgba(124,58,237,0.11);
    border: 1px solid rgba(139,92,246,0.24);
    font-size: 1.6rem;
    box-shadow: 0 0 30px rgba(124,58,237,0.10);
    animation: startupStar 1.8s ease-in-out infinite;
}}

.startup-title {{
    position: relative;
    z-index: 2;
    margin: 0;
    color: {TEXT};
    font-size: 1.55rem;
    font-weight: 780;
    letter-spacing: -0.025em;
}}

.startup-subtitle {{
    position: relative;
    z-index: 2;
    margin-top: 8px;
    color: {TEXT_SECONDARY};
    font-size: 0.88rem;
    line-height: 1.6;
}}

.startup-pills {{
    position: relative;
    z-index: 2;
    margin-top: 25px;
    display: flex;
    gap: 8px;
    justify-content: center;
    flex-wrap: wrap;
}}

.startup-pill {{
    padding: 7px 11px;
    border-radius: 999px;
    color: {TEXT_SECONDARY};
    background: rgba(124,58,237,0.055);
    border: 1px solid {BORDER};
    font-size: 0.69rem;
    font-weight: 650;
    letter-spacing: 0.025em;
}}

.startup-dots {{
    position: relative;
    z-index: 2;
    height: 12px;
    margin-top: 24px;
    display: flex;
    justify-content: center;
    align-items: center;
    gap: 7px;
}}

.startup-dot {{
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: {ACCENT};
    opacity: 0.3;
    animation: startupDot 1.2s ease-in-out infinite;
}}

.startup-dot:nth-child(2) {{
    animation-delay: 0.16s;
}}

.startup-dot:nth-child(3) {{
    animation-delay: 0.32s;
}}

@keyframes startupFade {{
    from {{
        opacity: 0;
        transform: translateY(10px) scale(0.985);
    }}
    to {{
        opacity: 1;
        transform: translateY(0) scale(1);
    }}
}}

@keyframes startupStar {{
    0%, 100% {{
        transform: scale(1) rotate(0deg);
        box-shadow: 0 0 22px rgba(124,58,237,0.10);
    }}
    50% {{
        transform: scale(1.08) rotate(8deg);
        box-shadow: 0 0 34px rgba(124,58,237,0.20);
    }}
}}

@keyframes startupDot {{
    0%, 100% {{
        transform: translateY(0) scale(0.9);
        opacity: 0.25;
    }}
    50% {{
        transform: translateY(-4px) scale(1.15);
        opacity: 1;
    }}
}}

@keyframes startupGlow {{
    0%, 100% {{
        opacity: 0.55;
        transform: scale(1);
    }}
    50% {{
        opacity: 0.95;
        transform: scale(1.08);
    }}
}}


/* =========================================================
   STREAMLIT UI CLEANUP
   ========================================================= */

/* Remove Streamlit's top white chrome without affecting the app */
header[data-testid="stHeader"],
[data-testid="stHeader"] {{
    background: transparent !important;
    height: 0rem !important;
    min-height: 0rem !important;
}}

[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"] {{
    display: none !important;
}}

/* Keep the full page on the selected theme */
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
section.main {{
    background: transparent !important;
}}

/* More breathing room at the top after hiding the header */
[data-testid="stMainBlockContainer"] {{
    padding-top: 1.4rem !important;
    padding-bottom: 7.5rem !important;
}}

/* Make every chat message readable in both themes */
div[data-testid="stChatMessage"],
div[data-testid="stChatMessage"] * {{
    color: {TEXT} !important;
}}

div[data-testid="stChatMessage"] p,
div[data-testid="stChatMessage"] li,
div[data-testid="stChatMessage"] span,
div[data-testid="stChatMessage"] code,
div[data-testid="stChatMessage"] strong,
div[data-testid="stChatMessage"] em {{
    color: {TEXT} !important;
}}

div[data-testid="stChatMessage"] code {{
    background: rgba(124,58,237,0.10) !important;
    border: 1px solid {BORDER} !important;
    border-radius: 6px !important;
    padding: 0.12rem 0.35rem !important;
}}

/* Chat message avatars */
div[data-testid="stChatMessage"] [data-testid="stChatMessageAvatarUser"],
div[data-testid="stChatMessage"] [data-testid="stChatMessageAvatarAssistant"] {{
    box-shadow: 0 6px 18px {SHADOW};
}}

/* Sources expander */
div[data-testid="stExpander"],
details[data-testid="stExpander"],
details {{
    background: {CARD} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 13px !important;
}}

div[data-testid="stExpander"] *,
details[data-testid="stExpander"] *,
details * {{
    color: {TEXT_SECONDARY} !important;
}}

details summary {{
    color: {TEXT_SECONDARY} !important;
}}

/* Remove the large white strip around Streamlit chat input */
[data-testid="stBottomBlockContainer"],
[data-testid="stBottomBlockContainer"] > div {{
    background: {BG} !important;
}}

[data-testid="stBottomBlockContainer"] {{
    border-top: 1px solid {BORDER} !important;
    padding-top: 0.75rem !important;
    padding-bottom: 0.85rem !important;
    box-shadow: 0 -16px 35px {SHADOW};
}}

/* Chat input itself */
div[data-testid="stChatInput"] {{
    background: transparent !important;
}}

div[data-testid="stChatInput"] > div {{
    background: {INPUT} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 17px !important;
    box-shadow: 0 10px 28px {SHADOW};
}}

div[data-testid="stChatInput"] textarea {{
    background: transparent !important;
    color: {TEXT} !important;
    caret-color: {ACCENT} !important;
}}

div[data-testid="stChatInput"] textarea::placeholder {{
    color: {TEXT_MUTED} !important;
    opacity: 1 !important;
}}

div[data-testid="stChatInput"] button {{
    color: {TEXT} !important;
}}

/* Sidebar native text/captions stay legible */
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] label {{
    color: {TEXT_SECONDARY} !important;
}}

/* Hide Streamlit deployment chrome */
.stDeployButton,
[data-testid="stAppDeployButton"] {{
    display: none !important;
}}


/* STREAMLIT */

#MainMenu {{
    visibility: hidden;
}}

footer {{
    visibility: hidden;
}}

</style>
""",
    unsafe_allow_html=True,
)


# =========================================================
# FOUNDRY LOCAL
# =========================================================

@st.cache_resource(show_spinner=False)
def initialize_rag():

    manager = FoundryLocalManager.instance

    if manager is None:

        config = Configuration(
            app_name="sidera"
        )

        FoundryLocalManager.initialize(
            config
        )

        manager = (
            FoundryLocalManager.instance
        )

    if manager is None:
        raise RuntimeError(
            "Foundry Local could not be initialized."
        )

    embedding_model = get_embedding_model(
        manager
    )

    embedding_client = (
        embedding_model.get_embedding_client()
    )

    chat_model = get_chat_model(
        manager
    )

    chat_client = (
        chat_model.get_chat_client()
    )

    return (
        embedding_model,
        embedding_client,
        chat_model,
        chat_client,
    )


# =========================================================
# LOAD MODELS
# =========================================================

startup_placeholder = st.empty()

startup_html = (
    '<div class="startup-wrap">'
    '<div class="startup-card">'
    '<div class="startup-star">✦</div>'
    '<div class="startup-title">Starting Sidera</div>'
    '<div class="startup-subtitle">'
    'Preparing your local knowledge engine and loading on-device AI models.'
    '</div>'
    '<div class="startup-pills">'
    '<span class="startup-pill">LOCAL / OFFLINE</span>'
    '<span class="startup-pill">EMBEDDINGS</span>'
    '<span class="startup-pill">PHI-3.5 MINI</span>'
    '</div>'
    '<div class="startup-dots">'
    '<span class="startup-dot"></span>'
    '<span class="startup-dot"></span>'
    '<span class="startup-dot"></span>'
    '</div>'
    '</div>'
    '</div>'
)

startup_placeholder.markdown(
    startup_html,
    unsafe_allow_html=True,
)

try:
    (
        embedding_model,
        embedding_client,
        chat_model,
        chat_client,
    ) = initialize_rag()

finally:
    startup_placeholder.empty()


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown(
        """
<div class="sidera-brand">
<span class="sidera-brand-star">✦</span>SIDERA
</div>

<div class="sidera-brand-subtitle">
Local Knowledge Intelligence
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown(
        """
<div class="status-card">

<div class="status-main">
● Local models ready
</div>

<div class="status-sub">
All systems operational
</div>

</div>
""",
        unsafe_allow_html=True,
    )

    st.divider()

    st.caption(
        "KNOWLEDGE ENGINE"
    )

    st.markdown(
        "**Microsoft Foundry Local**"
    )

    st.markdown("")

    st.caption(
        "EMBEDDING MODEL"
    )

    st.markdown(
        """
<div class="model-box">
qwen3-embedding-0.6b
</div>
""",
        unsafe_allow_html=True,
    )

    st.caption(
        "CHAT MODEL"
    )

    st.markdown(
        """
<div class="model-box">
phi-3.5-mini
</div>
""",
        unsafe_allow_html=True,
    )

    st.divider()

    st.caption(
        "RETRIEVAL"
    )

    st.markdown(
        "**Similarity threshold**"
    )

    st.markdown(
        """
<div class="model-box">
0.40
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown(
        "**Retrieval method**"
    )

    st.markdown(
        """
<div class="model-box">
Cosine Similarity
</div>
""",
        unsafe_allow_html=True,
    )

    st.divider()

    if st.button(
        "Clear conversation",
        use_container_width=True,
    ):

        st.session_state.messages = []

        st.rerun()

    st.caption(
        "100% Local • Offline"
    )


# =========================================================
# DOCUMENT UPLOAD / APP GATE
# =========================================================

document_ready = bool(
    st.session_state.active_document
)

uploaded_pdf = None
keep_existing_documents = False


# ---------------------------------------------------------
# FIRST SCREEN: REQUIRE A PDF
# ---------------------------------------------------------

if not document_ready:

    upload_gate_html = (
        '<div class="upload-gate">'
        '<div class="upload-gate-card">'
        '<div class="upload-gate-icon">✦</div>'
        '<div class="upload-gate-title">Start with your PDF</div>'
        '<div class="upload-gate-subtitle">'
        'Upload a document to build your local knowledge base. '
        'Sidera will extract the text, split it into chunks, '
        'generate embeddings, and index everything locally '
        'before the chat becomes available.'
        '</div>'
        '<div class="upload-gate-steps">'
        '<span class="upload-step">1 · UPLOAD</span>'
        '<span class="upload-step">2 · CHUNK</span>'
        '<span class="upload-step">3 · EMBED</span>'
        '<span class="upload-step">4 · CHAT</span>'
        '</div>'
        '<div class="upload-gate-note">'
        'Your document is processed locally through Sidera.'
        '</div>'
        '</div>'
        '</div>'
    )

    st.markdown(
        upload_gate_html,
        unsafe_allow_html=True,
    )

    uploaded_pdf = st.file_uploader(
        "Upload PDF",
        type=["pdf"],
        label_visibility="collapsed",
        key="initial_pdf_uploader",
        help=(
            "Upload a PDF to index it locally "
            "and unlock the chat."
        ),
    )


# ---------------------------------------------------------
# PDF READY: SHOW COMPACT DOCUMENT STATUS
# ---------------------------------------------------------

else:

    st.markdown(
        f"""
<div class="upload-active-card">
    <strong>● Document ready</strong>
    &nbsp; {st.session_state.active_document}
    &nbsp;·&nbsp;
    {st.session_state.active_document_chunks} chunks indexed
</div>
""",
        unsafe_allow_html=True,
    )

    with st.expander(
        "Change or add another PDF",
        expanded=False,
    ):

        uploaded_pdf = st.file_uploader(
            "Choose another PDF",
            type=["pdf"],
            key="replacement_pdf_uploader",
            help=(
                "Upload a new PDF to replace the current "
                "knowledge base, or keep the existing documents."
            ),
        )

        keep_existing_documents = st.checkbox(
            "Keep previously indexed documents",
            value=False,
            key="keep_existing_documents",
            help=(
                "Off: the new PDF replaces the current knowledge base. "
                "On: the new PDF is added alongside existing PDFs."
            ),
        )


# ---------------------------------------------------------
# INDEX UPLOADED PDF
# ---------------------------------------------------------

if uploaded_pdf is not None:

    uploaded_bytes = (
        uploaded_pdf.getvalue()
    )

    upload_hash = hashlib.sha256(
        uploaded_bytes
    ).hexdigest()

    if (
        upload_hash
        != st.session_state.indexed_upload_hash
    ):

        progress_bar = st.progress(
            0,
            text="Preparing document...",
        )

        def update_ingest_progress(
            completed: int,
            total: int,
        ) -> None:

            progress = (
                completed / total
                if total
                else 0
            )

            progress_bar.progress(
                progress,
                text=(
                    f"Indexing {uploaded_pdf.name} · "
                    f"{completed}/{total} chunks"
                ),
            )

        try:

            result = ingest_pdf_bytes(
                pdf_bytes=uploaded_bytes,
                source_name=uploaded_pdf.name,
                embedding_client=embedding_client,
                replace_knowledge_base=(
                    not keep_existing_documents
                ),
                progress_callback=(
                    update_ingest_progress
                ),
            )

            st.session_state.indexed_upload_hash = (
                upload_hash
            )

            st.session_state.active_document = (
                result["source_name"]
            )

            st.session_state.active_document_chunks = (
                result["chunks"]
            )

            # Yeni knowledge base ile eski konuşmayı karıştırma.
            st.session_state.messages = []

            progress_bar.empty()

            st.success(
                f"✓ {result['source_name']} indexed "
                f"({result['chunks']} chunks). "
                f"Opening chat..."
            )

            st.rerun()

        except Exception as error:

            progress_bar.empty()

            st.error(
                "Sidera couldn't index this PDF. "
                f"Error: {error}"
            )


# PDF hazır olmadan chat / hero / feature alanlarına geçme.
if not st.session_state.active_document:
    st.stop()


# =========================================================
# HERO
# =========================================================

st.markdown(
    """
<div class="hero">

<div class="hero-stars">
✦ ✧ ✦
</div>

<div class="hero-title">
Sidera
</div>

<div class="hero-tagline">
Local knowledge. Private intelligence.
</div>

<div class="hero-description">
Explore your local knowledge base through semantic retrieval
and document-grounded AI. Sidera retrieves relevant context
from your documents and generates answers using models running
locally through Microsoft Foundry Local.
</div>

<div class="badges">

<span class="badge badge-local">
● LOCAL / OFFLINE
</span>

<span class="badge badge-rag">
RAG POWERED
</span>

<span class="badge badge-private">
DOCUMENT GROUNDED
</span>

</div>

</div>
""",
    unsafe_allow_html=True,
)




# =========================================================
# EMPTY STATE
# =========================================================

if not st.session_state.messages:

    col1, col2, col3 = st.columns(
        3,
        gap="medium",
    )

    with col1:
        st.markdown(
            """
<div class="feature-card">
<div class="feature-symbol">✦</div>
<div class="feature-title">
Grounded Knowledge
</div>
<div class="feature-text">
Answers are generated from information retrieved
from your indexed local documents.
</div>
</div>
""",
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
<div class="feature-card">
<div class="feature-symbol">⌕</div>
<div class="feature-title">
Semantic Retrieval
</div>
<div class="feature-text">
Vector embeddings and cosine similarity identify
the most relevant context for each question.
</div>
</div>
""",
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            """
<div class="feature-card">
<div class="feature-symbol">◈</div>
<div class="feature-title">
Local Intelligence
</div>
<div class="feature-text">
Embedding and language models run locally through
Microsoft Foundry Local.
</div>
</div>
""",
            unsafe_allow_html=True,
        )

    st.markdown(
        """
<div class="welcome-card">
<div class="welcome-star">✦</div>
<div class="welcome-title">
Welcome to Sidera
</div>
<div class="welcome-text">
Ask anything about your local documents.
</div>
</div>
""",
        unsafe_allow_html=True,
    )


# =========================================================
# CHAT HISTORY
# =========================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )

        sources = message.get(
            "sources",
            [],
        )

        if sources:

            with st.expander(
                f"Sources · {len(sources)} retrieved chunks"
            ):

                for index, source in enumerate(
                    sources,
                    start=1,
                ):

                    source_name = source.get(
                        "source_name",
                        "Unknown source",
                    )

                    chunk_index = source.get(
                        "chunk_index",
                        "N/A",
                    )

                    score = float(
                        source.get(
                            "score",
                            0,
                        )
                    )

                    st.markdown(
                        f"""
**✦ Source {index}**

`{source_name}`

Chunk `{chunk_index}` · Similarity `{score:.4f}`
"""
                    )

                    if index != len(sources):
                        st.divider()


# =========================================================
# CHAT INPUT
# =========================================================

question = st.chat_input(
    "Ask Sidera about your documents..."
)


# =========================================================
# PROCESS QUESTION
# =========================================================

if question:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):

        with st.spinner(
            "Searching your knowledge base..."
        ):

            try:

                answer, sources = answer_query(
                    question,
                    embedding_client,
                    chat_client,
                )

                st.markdown(answer)

                if sources:

                    with st.expander(
                        f"Sources · {len(sources)} retrieved chunks"
                    ):

                        for index, source in enumerate(
                            sources,
                            start=1,
                        ):

                            source_name = source.get(
                                "source_name",
                                "Unknown source",
                            )

                            chunk_index = source.get(
                                "chunk_index",
                                "N/A",
                            )

                            score = float(
                                source.get(
                                    "score",
                                    0,
                                )
                            )

                            st.markdown(
                                f"""
**✦ Source {index}**

`{source_name}`

Chunk `{chunk_index}` · Similarity `{score:.4f}`
"""
                            )

                            if index != len(sources):
                                st.divider()

            except Exception as error:

                answer = (
                    "Sidera couldn't generate a response. "
                    f"Error: {error}"
                )

                sources = []

                st.error(answer)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources,
        }
    )


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
<div class="sidera-footer">
✦ SIDERA
&nbsp;&nbsp;•&nbsp;&nbsp;
Powered by Microsoft Foundry Local
&nbsp;&nbsp;•&nbsp;&nbsp;
Local RAG
</div>
""",
    unsafe_allow_html=True,
)