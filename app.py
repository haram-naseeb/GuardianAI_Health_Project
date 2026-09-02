"""GuardianAI — Premium Chat-First Emergency Health Assistant.

Streamlit UI for the GuardianAI multi-agent emergency response system.
Chat-first conversational interface with emergency detection pipeline.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import io
import os
import time
from typing import Any

import streamlit as st
from dotenv import load_dotenv
from google import genai
from PIL import Image

from app.agents.coordinator_agent import CoordinatorAgent

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------

st.set_page_config(
    page_title="GuardianAI — Your AI Health Guardian",
    page_icon="\U0001F6E1\ufe0f",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------

ACCENT = "#E91E8C"
ACCENT_RGB = "233, 30, 140"
# Gemini 3.6 uses the newer Interactions API.  Gemini 3.5 Flash is verified
# against this project's current ``models.generate_content`` implementation.
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")

CONVERSATIONAL_SYSTEM_PROMPT = """\
You are GuardianAI, a calm, empathetic AI health assistant.

Your role:
- Help users understand health concerns and injuries
- Provide practical first-aid guidance
- Ask focused clarifying questions about symptoms
- Help assess urgency and recommend next steps
- Stay concise — 2-4 sentences per response

Your personality:
- Calm and reassuring
- Empathetic but professional
- Safety-conscious — always prioritize the user's wellbeing
- Practical and actionable

Boundaries:
- You are NOT a substitute for professional medical care
- For potentially serious symptoms, recommend seeking medical attention
- For clearly non-health topics, politely redirect:
  "I'm focused on health and first-aid support. I'd be happy to help
  with a health concern — what are you experiencing?"

For greetings like "hi" or "hello":
  "Hi! I'm GuardianAI, your health assistant. I'm here to help with
  injuries, health concerns, and first-aid guidance. What are you
  experiencing today?"
"""

EMERGENCY_KEYWORDS = [
    "accident", "crash", "collision", "hit by", "crashed",
    "car accident", "road accident", "bike accident",
    "vehicle", "truck", "motorcycle",
    "hadsa", "takkar", "sadma",
    "bleeding heavily", "heavy bleeding", "won't stop bleeding",
    "unconscious", "not breathing", "no pulse",
    "heart attack", "chest pain", "stroke",
    "severe burn", "on fire",
    "choking", "can't breathe",
    "seizure", "convulsion",
    "fell from", "fall from height",
    "broken bone", "fracture",
    "emergency", "ambulance", "critical",
    "severe pain", "serious injury",
    "head injury", "spinal",
    "poisoning", "overdose",
    "stabbed", "gunshot",
    "electrocution", "drowning",
    "serious", "severe",
]

KARACHI_LAT = 24.8607
KARACHI_LON = 67.0011

HERO_IMAGE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "app", "assets", "guardian_hero.png",
)

# ---------------------------------------------------------
# CSS
# ---------------------------------------------------------

CSS = f"""
<style>
/* === HIDE STREAMLIN CHROME === */
#MainMenu, header, footer {{
    visibility: hidden;
    height: 0;
    padding: 0;
    margin: 0;
}}
.stDeployButton {{ display: none; }}

/* === DARK BACKGROUND === */
.stApp {{
    background: #0a0a0f;
    color: #e0e0e0;
}}

/* === SIDEBAR === */
section[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, #1a1a2e 0%, #16162a 100%);
    border-right: 1px solid rgba({ACCENT_RGB}, 0.2);
}}
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] .stMarkdown li,
section[data-testid="stSidebar"] .stMarkdown span,
section[data-testid="stSidebar"] label {{
    color: #e0e0e0;
}}
section[data-testid="stSidebar"] h2 {{
    background: linear-gradient(135deg, {ACCENT}, #ff6b9d);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}}

/* === RADIO NAVIGATION === */
div[role="radiogroup"] label {{
    padding: 0.6rem 1rem;
    border-radius: 8px;
    margin-bottom: 4px;
    transition: background 0.2s;
}}
div[role="radiogroup"] label:hover {{
    background: rgba({ACCENT_RGB}, 0.1) !important;
}}
div[role="radiogroup"] label[data-baseweb="radio"]:first-child {{
    background: rgba({ACCENT_RGB}, 0.15);
}}

/* === CHAT MESSAGES === */
div[data-testid="stChatMessage"] {{
    background: transparent !important;
    padding: 0 !important;
    margin-bottom: 0.5rem !important;
}}
div[data-testid="stChatMessage"] > div {{
    background: transparent !important;
}}
div[data-testid="stChatMessage"] [data-testid="chatAvatar"] {{
    width: 2rem;
    height: 2rem;
    min-width: 2rem;
    border-radius: 50%;
    font-size: 0.9rem;
}}
/* User avatar = burgundy */
div[data-testid="stChatMessage"]:has([data-testid="chatAvatar"]:nth-child(1))
    [data-testid="chatAvatar"] {{
    background: {ACCENT} !important;
}}

/* Chat message text */
div[data-testid="stChatMessage"] p {{
    color: #e0e0e0;
    line-height: 1.6;
    margin: 0;
}}
div[data-testid="stChatMessage"] strong {{
    color: #ffffff;
}}

/* === CHAT INPUT === */
.stChatInput {{
    border-top: 1px solid rgba(255,255,255,0.05);
    padding-top: 0.5rem;
}}
.stChatInput textarea {{
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba({ACCENT_RGB}, 0.3) !important;
    border-radius: 12px !important;
    color: #e0e0e0 !important;
    font-size: 0.95rem;
}}
.stChatInput textarea:focus {{
    border-color: {ACCENT} !important;
    box-shadow: 0 0 12px rgba({ACCENT_RGB}, 0.2) !important;
}}

/* === BUTTONS === */
.stButton > button {{
    border-radius: 8px;
    transition: all 0.2s;
}}

/* === COLOR-CODED QUICK ACTIONS === */
.st-key-qa_hospitals .stButton > button {{
    border-left: 3px solid #9C27B0 !important;
    background: linear-gradient(135deg, rgba(156,39,176,0.12) 0%, rgba(156,39,176,0.04) 100%) !important;
}}
.st-key-qa_hospitals .stButton > button:hover {{
    background: linear-gradient(135deg, rgba(156,39,176,0.22) 0%, rgba(156,39,176,0.08) 100%) !important;
    box-shadow: 0 0 12px rgba(156,39,176,0.2);
}}
.st-key-qa_firstaid .stButton > button {{
    border-left: 3px solid #2196F3 !important;
    background: linear-gradient(135deg, rgba(33,150,243,0.12) 0%, rgba(33,150,243,0.04) 100%) !important;
}}
.st-key-qa_firstaid .stButton > button:hover {{
    background: linear-gradient(135deg, rgba(33,150,243,0.22) 0%, rgba(33,150,243,0.08) 100%) !important;
    box-shadow: 0 0 12px rgba(33,150,243,0.2);
}}
.st-key-qa_contacts .stButton > button {{
    border-left: 3px solid #4CAF50 !important;
    background: linear-gradient(135deg, rgba(76,175,80,0.12) 0%, rgba(76,175,80,0.04) 100%) !important;
}}
.st-key-qa_contacts .stButton > button:hover {{
    background: linear-gradient(135deg, rgba(76,175,80,0.22) 0%, rgba(76,175,80,0.08) 100%) !important;
    box-shadow: 0 0 12px rgba(76,175,80,0.2);
}}
.st-key-qa_tips .stButton > button {{
    border-left: 3px solid #009688 !important;
    background: linear-gradient(135deg, rgba(0,150,136,0.12) 0%, rgba(0,150,136,0.04) 100%) !important;
}}
.st-key-qa_tips .stButton > button:hover {{
    background: linear-gradient(135deg, rgba(0,150,136,0.22) 0%, rgba(0,150,136,0.08) 100%) !important;
    box-shadow: 0 0 12px rgba(0,150,136,0.2);
}}

/* === FILE UPLOADER === */
[data-testid="stFileUploader"] {{
    background: rgba(255,255,255,0.03) !important;
    border: 1px dashed rgba({ACCENT_RGB}, 0.3) !important;
    border-radius: 12px;
    padding: 0.5rem;
}}

/* === GLASS CARD === */
.glass-card {{
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 1.2rem;
    backdrop-filter: blur(10px);
    margin-bottom: 0.8rem;
}}

/* === GUARDIAN HEADER === */
.guardian-header-bar {{
    background: linear-gradient(135deg,
        rgba({ACCENT_RGB},0.15) 0%,
        rgba(100,50,150,0.1) 100%);
    border: 1px solid rgba({ACCENT_RGB},0.2);
    border-radius: 12px;
    padding: 0.8rem 1.5rem;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
}}

/* === HEADER BUTTONS === */
.header-btn {{
    background: rgba({ACCENT_RGB},0.15);
    border: 1px solid rgba({ACCENT_RGB},0.3);
    color: {ACCENT};
    padding: 0.4rem 1rem;
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.2s;
    text-decoration: none;
}}
.header-btn:hover {{
    background: rgba({ACCENT_RGB},0.25);
    box-shadow: 0 0 10px rgba({ACCENT_RGB},0.2);
}}

/* === HEALTH GAUGE (Semicircular Arc) === */
.health-gauge {{
    position: relative;
    width: 140px;
    height: 75px;
    margin: 0.8rem auto 0;
    overflow: hidden;
}}
.health-gauge-bg {{
    position: absolute;
    width: 140px;
    height: 140px;
    border-radius: 50%;
    background: conic-gradient(
        from 270deg,
        rgba(255,255,255,0.08) 0deg,
        rgba(255,255,255,0.08) 180deg,
        transparent 180deg
    );
    top: 0;
    left: 0;
}}
.health-gauge-fill {{
    position: absolute;
    width: 140px;
    height: 140px;
    border-radius: 50%;
    top: 0;
    left: 0;
}}
.gauge-low {{
    background: conic-gradient(
        from 270deg,
        #4CAF50 0deg, #4CAF50 45deg,
        transparent 45deg
    );
}}
.gauge-moderate {{
    background: conic-gradient(
        from 270deg,
        #FF9800 0deg, #FF9800 90deg,
        transparent 90deg
    );
}}
.gauge-high {{
    background: conic-gradient(
        from 270deg,
        #f44336 0deg, #f44336 135deg,
        transparent 135deg
    );
}}
.gauge-critical {{
    background: conic-gradient(
        from 270deg,
        #E91E63 0deg, #E91E63 171deg,
        transparent 171deg
    );
    animation: pulse-gauge 2s infinite;
}}
.gauge-monitoring {{
    background: conic-gradient(
        from 270deg,
        rgba(255,255,255,0.12) 0deg,
        rgba(255,255,255,0.12) 90deg,
        transparent 90deg
    );
}}
@keyframes pulse-gauge {{
    0%, 100% {{ opacity: 1; }}
    50% {{ opacity: 0.7; }}
}}
.health-gauge-inner {{
    position: absolute;
    width: 100px;
    height: 100px;
    border-radius: 50%;
    background: #111118;
    top: 20px;
    left: 20px;
    z-index: 1;
}}
.health-gauge-divider {{
    position: absolute;
    bottom: 0;
    left: 0;
    right: 0;
    height: 1px;
    background: rgba(255,255,255,0.1);
}}
.health-gauge-label {{
    position: absolute;
    bottom: 6px;
    left: 0;
    right: 0;
    text-align: center;
    font-weight: 700;
    font-size: 0.85rem;
    z-index: 2;
    text-transform: uppercase;
    letter-spacing: 1px;
}}
.health-gauge-label.gauge-low {{ color: #4CAF50; }}
.health-gauge-label.gauge-moderate {{ color: #FF9800; }}
.health-gauge-label.gauge-high {{ color: #f44336; }}
.health-gauge-label.gauge-critical {{ color: #E91E63; }}
.health-gauge-label.gauge-monitoring {{ color: rgba(255,255,255,0.4); }}

/* === EMERGENCY SECTION === */
.emergency-box {{
    background: rgba(244,67,54,0.08);
    border: 1px solid rgba(244,67,54,0.25);
    border-radius: 12px;
    padding: 1rem;
    margin-top: 1.5rem;
    text-align: center;
}}

/* === ONLINE INDICATOR === */
.online-dot {{
    width: 8px;
    height: 8px;
    background: #4CAF50;
    border-radius: 50%;
    display: inline-block;
    margin-right: 6px;
    animation: pulse-green 2s infinite;
}}
@keyframes pulse-green {{
    0%, 100% {{ opacity: 1; }}
    50% {{ opacity: 0.5; }}
}}

/* === RIGHT PANEL === */
.rp-section {{
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 12px;
    padding: 1rem;
    margin-bottom: 0.6rem;
}}
.rp-title {{
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: rgba(255,255,255,0.4);
    margin-bottom: 0.5rem;
}}
.tip-card {{
    background: rgba({ACCENT_RGB},0.05);
    border: 1px solid rgba({ACCENT_RGB},0.15);
    border-radius: 8px;
    padding: 0.7rem;
    font-size: 0.85rem;
    color: #ccc;
    line-height: 1.5;
}}

/* === EXPANDERS === */
.stExpander {{
    background: rgba(255,255,255,0.02) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 8px !important;
}}
.stExpander summary span {{
    color: #e0e0e0 !important;
    font-size: 0.85rem !important;
}}

/* === HOSPITAL CARD IN CHAT === */
.hospital-card {{
    background: rgba(76,175,80,0.08);
    border: 1px solid rgba(76,175,80,0.25);
    border-radius: 10px;
    padding: 0.8rem 1rem;
    margin: 0.5rem 0;
}}
.hospital-card strong {{ color: #81C784; }}

/* === SELECT === */
.stSelectbox label {{
    color: #aaa !important;
    font-size: 0.8rem !important;
}}

/* === SPINNER === */
.stSpinner > div {{
    border-top-color: {ACCENT} !important;
}}

/* === BACKGROUND GRADIENT === */
.stApp {{
    background: linear-gradient(160deg, #0a0a0f 0%, #0d0d1a 40%, #0f0a14 100%) !important;
}}

/* === USER CHAT BUBBLE === */
div[data-testid="stChatMessage"]:has([data-testid="chatAvatar"]:nth-child(1)) {{
    background: rgba({ACCENT_RGB},0.06) !important;
    border-radius: 12px !important;
    padding: 0.6rem 0.8rem !important;
    border-left: 2px solid {ACCENT} !important;
}}

/* === ASSISTANT CHAT BUBBLE === */
div[data-testid="stChatMessage"] {{
    margin-bottom: 0.8rem !important;
}}
div[data-testid="stChatMessage"] p {{
    font-size: 0.92rem;
    line-height: 1.7;
}}

/* === HEADER GLOW === */
.guardian-header-bar {{
    box-shadow: 0 4px 20px rgba({ACCENT_RGB},0.1);
}}

/* === RIGHT PANEL GLOW === */
.rp-section {{
    border-color: rgba({ACCENT_RGB},0.08);
}}
.rp-section:hover {{
    border-color: rgba({ACCENT_RGB},0.15);
    transition: border-color 0.3s;
}}

/* === FOOTER BAR === */
.guardian-footer {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.6rem 1.5rem;
    margin-top: 0.5rem;
    border-top: 1px solid rgba(255,255,255,0.06);
    gap: 1rem;
}}
.guardian-footer .footer-copy {{
    font-size: 0.7rem;
    color: rgba(255,255,255,0.3);
    white-space: nowrap;
}}
.guardian-footer .footer-warn {{
    display: flex;
    align-items: center;
    gap: 6px;
    background: rgba({ACCENT_RGB},0.06);
    border: 1px solid rgba({ACCENT_RGB},0.12);
    border-radius: 8px;
    padding: 0.35rem 0.8rem;
    font-size: 0.7rem;
    color: rgba(255,255,255,0.45);
    flex: 1;
    justify-content: center;
}}

/* === CHAT TIMESTAMP === */
.chat-timestamp {{
    font-size: 0.65rem;
    color: rgba(255,255,255,0.25);
    margin-top: 3px;
}}

/* === CHAT SUBTITLE === */
.chat-subtitle {{
    font-size: 0.85rem;
    color: rgba(255,255,255,0.45);
    margin-top: -0.3rem;
    margin-bottom: 0.5rem;
}}

/* === 3-PANEL LAYOUT === */
.st-key-chat_scroll {{
    height: calc(100vh - 320px) !important;
    min-height: 260px !important;
    flex-direction: column-reverse !important;
    justify-content: flex-end !important;
    overflow-y: auto !important;
    padding-right: 0.6rem;
    min-height: 0 !important;
}}
.st-key-chat_scroll > div {{
    min-height: 0 !important;
    overflow-y: auto !important;
}}
.st-key-chat_scroll::-webkit-scrollbar {{ width: 6px; }}
.st-key-chat_scroll::-webkit-scrollbar-thumb {{
    background: rgba({ACCENT_RGB},0.25);
    border-radius: 3px;
}}
.st-key-rp_root {{ max-height: calc(100vh - 250px); overflow-y: auto; }}
.st-key-rp_health, .st-key-rp_actions {{
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba({ACCENT_RGB},0.08);
    border-radius: 12px; padding: 1rem; margin-bottom: 0.6rem;
}}
section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] > div > .stVerticalBlock {{
    display: flex; flex-direction: column; min-height: calc(100vh - 7rem);
}}
.st-key-sidebar_hero {{ position: sticky; bottom: 0; text-align: center; padding: 0.5rem 0; background: linear-gradient(180deg, rgba(10,10,15,0) 0%, #0e0e18 30%); z-index: 1; }}
.st-key-sidebar_hero img {{
    max-width: 180px; border-radius: 12px;
    border: 1px solid rgba({ACCENT_RGB},0.3);
    box-shadow: 0 0 30px rgba({ACCENT_RGB},0.2), 0 0 60px rgba({ACCENT_RGB},0.1);
}}
[data-testid="stBottomBlockContainer"] {{
    background: linear-gradient(180deg, rgba(10,10,15,0) 0%, #0a0a0f 40%);
}}

/* === GUARDIANAI DASHBOARD SHELL === */
/* These rules deliberately target Streamlit's stable test ids so the layout
   remains a dashboard rather than falling back to the default Streamlit UI. */
.block-container {{
    max-width: 1480px !important;
    padding: .55rem 1.05rem 4.25rem !important;
}}
section[data-testid="stSidebar"] {{
    min-width: 238px !important;
}}
section[data-testid="stSidebar"] [data-testid="stSidebarContent"] {{
    padding: 0.5rem 0.65rem 1rem;
}}
.st-key-chat_shell {{
    background: linear-gradient(145deg, rgba(20,23,38,.96), rgba(8,11,20,.94));
    border: 1px solid rgba(166,182,225,.27);
    border-radius: 18px;
    padding: 1rem .9rem .65rem;
    box-shadow: inset 0 1px 0 rgba(255,255,255,.035), 0 16px 45px rgba(0,0,0,.18);
}}
.st-key-rp_root {{
    background: transparent;
    max-height: calc(100vh - 165px);
    padding-right: .1rem;
}}
.st-key-rp_health, .st-key-rp_actions, .st-key-rp_tips {{
    background: linear-gradient(145deg, rgba(23,27,43,.93), rgba(12,16,29,.92));
    border: 1px solid rgba(166,182,225,.22);
    border-radius: 15px;
    padding: .8rem .8rem;
    margin-bottom: .7rem;
}}
.guardian-topbar {{
    display: flex; align-items:center; justify-content:space-between; gap:1rem;
    padding: .15rem 0 .7rem;
}}
.guardian-wordmark {{ color:#f8f8fc; font-size:1.45rem; font-weight:800; letter-spacing:-.04em; }}
.guardian-wordmark span {{ color:{ACCENT}; }}
.guardian-tagline {{ color:#b8bdc9; font-size:.78rem; margin-top:-.15rem; }}
.top-pills {{ display:flex; align-items:center; gap:.6rem; flex-wrap:wrap; justify-content:flex-end; }}
.top-pill {{
    color:#ececf3; border:1px solid rgba(166,182,225,.2); border-radius:999px;
    background:rgba(10,13,24,.65); padding:.58rem .95rem; font-size:.82rem;
}}
.top-pill-accent {{ color:#ff6ab5; border-color:rgba({ACCENT_RGB},.35); }}
.assistant-heading {{ display:flex; align-items:center; justify-content:space-between; gap:1rem; }}
.assistant-heading h2 {{ font-size:1.22rem; margin:0; color:#f6f6f8; }}
.assistant-heading h2 span {{ color:{ACCENT}; }}
.online-badge {{ color:#e8ebf1; border:1px solid rgba(166,182,225,.2); background:#0b0f1a; border-radius:999px; padding:.45rem .75rem; font-size:.76rem; }}
.online-badge i {{ width:9px; height:9px; display:inline-block; background:#28d87a; border-radius:50%; margin-right:7px; }}
.chat-subtitle {{ margin:.25rem 0 .8rem; }}
.st-key-chat_scroll {{ height:calc(100vh - 355px) !important; min-height:280px !important; overflow-y:auto !important; }}
.st-key-chat_scroll > div {{ min-height:0 !important; overflow-y:auto !important; }}
div[data-testid="stChatMessage"] {{ max-width:86%; padding:.55rem .7rem !important; border:1px solid rgba(166,182,225,.15); border-radius:16px !important; background:linear-gradient(145deg, rgba(27,32,49,.92), rgba(17,21,34,.92)) !important; }}
div[data-testid="stChatMessage"][data-testid="stChatMessage"]:has([data-testid="stChatMessageContent"]), div[data-testid="stChatMessage"] {{ margin-bottom:.75rem !important; }}
[data-testid="stChatMessageContent"] {{ color:#e9e9ef; }}
.stChatInput textarea {{ min-height:3.5rem !important; padding:.8rem 1rem !important; background:#131827 !important; }}
.st-key-main_chat_input {{ margin-top: .85rem; }}
.st-key-main_chat_input textarea {{ min-height: 3.8rem !important; }}
section.main {{ overflow: hidden; }}
[data-testid="stMainBlockContainer"] {{
    max-height: 100vh;
    overflow: hidden;
    padding-bottom: .75rem !important;
}}
.st-key-rp_root {{ overflow: hidden !important; }}
.st-key-left_nav {{
    background: linear-gradient(180deg, rgba(20,23,38,.96), rgba(11,13,24,.96));
    border: 1px solid rgba(166,182,225,.18);
    border-radius: 15px;
    padding: .45rem .4rem .8rem;
    min-height: calc(100vh - 1.25rem);
}}
.st-key-left_nav .stButton > button {{
    text-align: left; min-height: 2.35rem; color: #eef0f7;
    background: transparent; border: 1px solid transparent;
    border-radius: 9px; font-size: .82rem;
}}
.st-key-left_nav .stButton > button:hover {{
    background: rgba({ACCENT_RGB},.12); border-color: rgba({ACCENT_RGB},.36);
}}
section[data-testid="stSidebar"] .stButton > button {{
    text-align:left; color:#eff0f4; border:1px solid transparent; background:transparent;
    min-height:2.7rem;
}}
section[data-testid="stSidebar"] .stButton > button:hover {{
    border-color:rgba({ACCENT_RGB},.45); background:rgba({ACCENT_RGB},.12);
}}
@media (max-width: 900px) {{
    .block-container {{ padding: .65rem .75rem 4.5rem !important; }}
    .guardian-topbar {{ align-items:flex-start; flex-direction:column; }}
    .top-pills {{ justify-content:flex-start; }}
    .st-key-chat_scroll {{ height:440px !important; }}
}}
</style>
"""

# ---------------------------------------------------------
# RESOURCE LOADERS
# ---------------------------------------------------------


@st.cache_resource(show_spinner="Initializing GuardianAI agents...")
def get_coordinator() -> CoordinatorAgent:
    load_dotenv()
    return CoordinatorAgent()


@st.cache_resource
def get_conversational_client() -> genai.Client:
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not set")
    return genai.Client(api_key=api_key)


@st.cache_resource
def _get_location_agent():
    from location_agent.agent import LocationAgent
    return LocationAgent()


# ---------------------------------------------------------
# SESSION STATE DEFAULTS
# ---------------------------------------------------------


def init_state():
    defaults = {
        "messages": [
            {
                "role": "assistant",
                "content": (
                    "Hello! I'm GuardianAI, your health assistant. "
                    "I'm here to help with injuries, health concerns, "
                    "and first-aid guidance. What are you experiencing "
                    "today?"
                ),
            }
        ],
        "last_result": None,
        "severity": None,
        "last_elapsed": None,
        "is_emergency": False,
        "latitude": KARACHI_LAT,
        "longitude": KARACHI_LON,
        "nav_page": "Assistant",
        "show_report": False,
        "analysis_summary": None,
        "show_right_panel": True,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# ---------------------------------------------------------
# MESSAGE HELPER
# ---------------------------------------------------------


def _append_msg(role: str, content: str, image=None):
    """Append a chat message with timestamp."""
    msg: dict[str, Any] = {
        "role": role,
        "content": content,
        "time": time.strftime("%I:%M %p").lstrip("0"),
    }
    if image is not None:
        msg["image"] = image
    st.session_state.messages.append(msg)


# ---------------------------------------------------------
# MESSAGE ROUTING
# ---------------------------------------------------------


def is_emergency_message(text: str) -> bool:
    if not text:
        return False
    lower = text.lower()
    for kw in EMERGENCY_KEYWORDS:
        if kw in lower:
            return True
    return False


def route_message(text: str, has_image: bool) -> str:
    if has_image:
        return "emergency"
    if not text.strip():
        return "empty"
    if is_emergency_message(text):
        return "emergency"
    return "conversation"


# ---------------------------------------------------------
# CONVERSATIONAL GEMINI
# ---------------------------------------------------------


def call_conversational_gemini(user_message: str) -> str:
    client = get_conversational_client()

    history = []
    for m in st.session_state.messages[-20:]:
        if m["role"] in ("user", "assistant"):
            role = "model" if m["role"] == "assistant" else "user"
            history.append({"role": role, "parts": [{"text": m["content"]}]})

    if st.session_state.get("analysis_summary"):
        context_note = (
            "\n\n[Context: A previous emergency analysis found: "
            f"{st.session_state.analysis_summary} "
            "Use this context when relevant.]"
        )
        history.append({
            "role": "model",
            "parts": [{"text": context_note}],
        })

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[
                {"role": "user", "parts": [{"text": CONVERSATIONAL_SYSTEM_PROMPT}]},
                {"role": "model", "parts": [{"text": "Understood. I'm GuardianAI, ready to help with health concerns."}]},
                *history,
                {"role": "user", "parts": [{"text": user_message}]},
            ],
        )
        return response.text.strip() if response.text else "I'm here to help. Could you tell me more about what you're experiencing?"
    except Exception as exc:
        # Keep technical provider errors out of the emergency-facing chat UI.
        # The exception class is retained in the server log for debugging.
        print(f"Gemini conversational request failed: {type(exc).__name__}: {exc}")
        return (
            "I'm temporarily unable to reach the medical guidance service. "
            "Please try again in a moment. If this is urgent, call 1122 "
            "or seek immediate medical help."
        )


# ---------------------------------------------------------
# EMERGENCY PIPELINE
# ---------------------------------------------------------


def run_emergency_pipeline(
    description: str,
    image: Image.Image | None,
):
    coordinator = get_coordinator()
    start = time.time()
    result = coordinator.handle_case(
        user_description=description or "",
        image_input=image,
        latitude=st.session_state.latitude,
        longitude=st.session_state.longitude,
    )
    elapsed = time.time() - start

    if result.medical_result:
        sev = str(result.medical_result.get("severity", "")).upper()
        if sev in ("LOW", "MODERATE", "HIGH", "CRITICAL"):
            st.session_state.severity = sev

    st.session_state.last_result = result
    st.session_state.last_elapsed = elapsed
    st.session_state.is_emergency = True

    guardian_msg = _build_emergency_response(result, elapsed)
    _append_msg("assistant", guardian_msg)

    parts = []
    if result.medical_result:
        parts.append(
            f"Severity: {result.medical_result.get('severity', 'unknown')}"
        )
    if result.case_type:
        parts.append(f"Case: {result.case_type}")
    if result.knowledge_result and result.knowledge_result.get("answer"):
        parts.append(
            f"Guidance: {result.knowledge_result['answer'][:200]}"
        )
    st.session_state.analysis_summary = " | ".join(parts)


def _build_emergency_response(result, elapsed: float) -> str:
    parts = []

    if result.status == "complete":
        parts.append(
            "I've completed a full emergency assessment "
            f"({elapsed:.1f}s). Here's what I found:"
        )
    elif result.status == "partial":
        parts.append(
            "I've run the emergency assessment. "
            "Some agents encountered issues, but here's what I found:"
        )
    else:
        parts.append(
            "The emergency assessment encountered difficulties. "
            "Here's what's available:"
        )

    if result.medical_result:
        sev = result.medical_result.get("severity", "Unknown")
        emergency = result.medical_result.get("emergency", False)
        reasons = result.medical_result.get("reason", [])
        parts.append(f"\n**Medical Assessment: Severity {sev}**")
        if reasons:
            for r in reasons[:3]:
                parts.append(f"- {r}")
        if emergency:
            parts.append(
                "\n\u26a0\ufe0f **This may require immediate "
                "medical attention.**"
            )

    if (
        result.knowledge_result
        and result.knowledge_result.get("guidance_available")
    ):
        answer = result.knowledge_result.get("answer", "")
        if answer:
            parts.append(f"\n**First-Aid Guidance:**\n{answer}")

    if result.location_result:
        loc = result.location_result
        if loc.get("status") == "success":
            fac = loc.get("recommended_facility", {})
            parts.append(
                f"\n**Nearest Hospital:** {fac.get('name', 'Unknown')}"
            )
            dist = fac.get("distance_km", "?")
            travel = fac.get("travel_time_minutes", "?")
            parts.append(
                f"Distance: {dist} km | Estimated travel: {travel} min"
            )
            addr = fac.get("address", "")
            if addr:
                parts.append(f"Address: {addr}")

    if result.errors:
        parts.append(
            f"\n*Note: {len(result.errors)} agent(s) encountered issues.*"
        )

    parts.append(
        "\nYou can ask me follow-up questions, request a detailed "
        "report, or ask me to find more nearby hospitals."
    )

    return "\n".join(parts)


# ---------------------------------------------------------
# REPORT BUILDER
# ---------------------------------------------------------


def generate_report_text() -> str:
    result = st.session_state.get("last_result")
    if not result:
        return ""

    lines = ["# GuardianAI Health Report\n"]
    lines.append(f"**Status:** {result.status.upper()}")
    lines.append(
        f"**Case Type:** {result.case_type.replace('_', ' ').title()}"
    )
    lines.append(f"**Classifier:** {result.classifier_source}")

    elapsed = st.session_state.get("last_elapsed", 0)
    if elapsed:
        lines.append(f"**Analysis Time:** {elapsed:.1f}s")

    if result.medical_result:
        lines.append("\n## Medical Assessment")
        m = result.medical_result
        lines.append(f"**Severity:** {m.get('severity', 'N/A')}")
        lines.append(f"**Emergency:** {m.get('emergency', 'N/A')}")
        for r in m.get("reason", []):
            lines.append(f"- {r}")

    if result.vision_result and not result.vision_result.get(
        "vision_unavailable"
    ):
        lines.append("\n## Vision Analysis")
        v = result.vision_result
        lines.append(f"**Scene:** {v.get('scene_type', 'N/A')}")
        lines.append(
            f"**Visible Bleeding:** {v.get('visible_bleeding', 'N/A')}"
        )
        for obs in v.get("observations", []):
            lines.append(f"- {obs}")

    if (
        result.knowledge_result
        and result.knowledge_result.get("guidance_available")
    ):
        lines.append("\n## First-Aid Guidance")
        lines.append(result.knowledge_result.get("answer", ""))
        steps = result.knowledge_result.get("steps", [])
        if steps:
            lines.append("\n**Steps:**")
            for s in steps:
                lines.append(f"- {s}")

    if result.location_result:
        loc = result.location_result
        if loc.get("status") == "success":
            lines.append("\n## Recommended Hospital")
            fac = loc.get("recommended_facility", {})
            lines.append(f"**{fac.get('name', 'Unknown')}**")
            lines.append(
                f"Distance: {fac.get('distance_km', '?')} km | "
                f"Travel: {fac.get('travel_time_minutes', '?')} min"
            )
            lines.append(f"Address: {fac.get('address', 'N/A')}")

    if result.errors:
        lines.append("\n## Errors")
        for err in result.errors:
            lines.append(f"- {err}")

    return "\n".join(lines)


# ---------------------------------------------------------
# FIND NEARBY HOSPITALS
# ---------------------------------------------------------


def run_find_hospitals() -> str:
    agent = _get_location_agent()
    result = agent.find_best_facility(
        latitude=st.session_state.latitude,
        longitude=st.session_state.longitude,
    )
    d = result.to_dict()

    if d.get("status") == "success":
        fac = d.get("recommended_facility", {})
        parts = [
            "**Nearby Hospitals:**\n",
            f"**Recommended: {fac.get('name', 'Unknown')}**",
            f"Distance: {fac.get('distance_km', '?')} km | "
            f"Travel: ~{fac.get('travel_time_minutes', '?')} min "
            f"({fac.get('route_status', 'estimated')})",
        ]
        addr = fac.get("address", "")
        if addr:
            parts.append(f"Address: {addr}")

        alts = d.get("alternatives", [])
        if alts:
            parts.append(f"\n*{len(alts)} alternative(s) nearby:*")
            for a in alts[:3]:
                parts.append(
                    f"- **{a.get('name', '?')}** "
                    f"({a.get('distance_km', '?')} km)"
                )

        return "\n".join(parts)

    return (
        "I couldn't find nearby hospitals. "
        f"Error: {d.get('message', 'Unknown')}"
    )


# ---------------------------------------------------------
# UI: HEADER
# ---------------------------------------------------------


def _toggle_right_panel():
    st.session_state.show_right_panel = not st.session_state.show_right_panel


def _find_hospitals_from_header():
    with st.spinner("Searching..."):
        hospital_msg = run_find_hospitals()
    _append_msg("assistant", hospital_msg)


def render_header():
    brand, hospital, language, panel = st.columns([4.2, 2.2, 1.4, .45])
    with brand:
        st.markdown(
            '<div class="guardian-topbar"><div>'
            '<div class="guardian-wordmark">🛡️ Guardian<span>AI</span></div>'
            '<div class="guardian-tagline">Your AI Health Guardian</div>'
            '</div></div>', unsafe_allow_html=True,
        )
    with hospital:
        st.button("📍  Nearby Hospitals", key="header_hospitals",
                  on_click=_find_hospitals_from_header,
                  use_container_width=True)
    with language:
        st.markdown('<div class="top-pill">🌐 &nbsp;English ▾</div>',
                    unsafe_allow_html=True)
    with panel:
        st.button("◀" if st.session_state.show_right_panel else "▶",
                  key="header_toggle_panel", on_click=_toggle_right_panel,
                  use_container_width=True)


# ---------------------------------------------------------
# UI: SIDEBAR
# ---------------------------------------------------------


def render_sidebar(container=None):
    """Render persistent in-page navigation (not the collapsible sidebar)."""
    if container is None:
        container = st.sidebar
    with container:
        st.markdown('<div class="guardian-wordmark" style="font-size:1.25rem; padding:.55rem .3rem .85rem;">🛡️ Guardian<span>AI</span></div>', unsafe_allow_html=True)
        pages = [
            ("💬  Assistant", "Assistant"), ("📍  Hospitals", "Hospitals"),
            ("📋  Reports", "Reports"), ("👤  Profile", "Profile"),
            ("⚙️  Settings", "Settings"), ("ⓘ  About", "About"),
        ]
        for label, page in pages:
            if st.button(label, key=f"nav_{page}", use_container_width=True,
                         type="primary" if st.session_state.nav_page == page else "secondary"):
                st.session_state.nav_page = page
                st.rerun()

        st.markdown('<div style="height:.8rem"></div>', unsafe_allow_html=True)
        with st.expander("⚙️  Location settings"):
            st.number_input(
                "Latitude",
                min_value=-90.0,
                max_value=90.0,
                value=st.session_state.latitude,
                step=0.0001,
                format="%.4f",
                key="sidebar_lat",
                on_change=lambda: st.session_state.update(
                    {"latitude": st.session_state.sidebar_lat}
                ),
            )
            st.number_input(
                "Longitude",
                min_value=-180.0,
                max_value=180.0,
                value=st.session_state.longitude,
                step=0.0001,
                format="%.4f",
                key="sidebar_lon",
                on_change=lambda: st.session_state.update(
                    {"longitude": st.session_state.sidebar_lon}
                ),
            )
            if st.button("\U0001F4CD Use Karachi defaults"):
                st.session_state.latitude = KARACHI_LAT
                st.session_state.longitude = KARACHI_LON
                st.rerun()

        st.markdown(
            '<div class="emergency-box">'
            '<p style="color:#f44336;font-weight:700;margin:0;'
            'font-size:0.9rem;">\U0001F6A8 EMERGENCY</p>'
            '<p style="color:rgba(255,255,255,0.5);font-size:0.75rem;'
            'margin:0.3rem 0;">In a life-threatening situation?</p>'
            '<div style="display:inline-block;margin-top:0.5rem;color:#ff9abf;font-weight:700;">📞 Call emergency: 1122</div>'
            "</div>",
            unsafe_allow_html=True,
        )

        st.markdown("")

        st.markdown(
            '<p style="text-align:center;color:rgba(255,255,255,0.3);'
            'font-size:0.7rem;">Built for the Alibaba Hackathon</p>',
            unsafe_allow_html=True,
        )

        if os.path.isfile(HERO_IMAGE):
            with st.container(border=False, key="sidebar_hero"):
                st.image(HERO_IMAGE, use_container_width=True)


# ---------------------------------------------------------
# UI: CHAT AREA
# ---------------------------------------------------------


def render_chat_area():
    with st.container(border=False, key="chat_shell"):
        st.markdown(
            '<div class="assistant-heading"><div><h2>Guardian<span>AI</span> Assistant</h2>'
            '<p class="chat-subtitle">How can I help you today?</p></div>'
            '<div class="online-badge"><i></i>Online</div></div>',
            unsafe_allow_html=True,
        )
        messages = st.session_state.messages
        with st.container(height=460, border=False, key="chat_scroll"):
            for msg in reversed(messages):
                avatar = "🛡️" if msg["role"] == "assistant" else "👤"
                with st.chat_message(msg["role"], avatar=avatar):
                    if msg.get("image"):
                        st.image(msg["image"], width=300)
                    st.markdown(msg["content"])
                    if msg.get("time"):
                        st.markdown(
                            f'<div class="chat-timestamp">{msg["time"]}</div>',
                            unsafe_allow_html=True,
                        )

    if st.session_state.get("show_report"):
        report = generate_report_text()
        if report:
            st.markdown("---")
            st.markdown(report)
            st.download_button(
                "\U0001F4E5 Download Report",
                report,
                file_name="guardianai_report.md",
                mime="text/markdown",
            )
            if st.button("\u2716 Close Report"):
                st.session_state.show_report = False
                st.rerun()


# ---------------------------------------------------------
# UI: RIGHT PANEL
# ---------------------------------------------------------


def render_right_panel():
    with st.container(border=False, key="rp_root"):
        _render_health_status()
        _render_quick_actions()
        _render_tips()

        if st.session_state.is_emergency and st.session_state.last_result:
            _render_analysis_badge()


def _render_health_status():
    with st.container(border=False, key="rp_health"):
        sev = st.session_state.severity
        if sev:
            gauge_class = f"gauge-{sev.lower()}"
            label = sev
        else:
            gauge_class = "gauge-monitoring"
            label = "Monitoring"

        st.markdown(
            f'<div class="rp-title">\u2764\ufe0f Health Status</div>'
            f'<div class="health-gauge">'
            f'<div class="health-gauge-bg"></div>'
            f'<div class="health-gauge-fill {gauge_class}"></div>'
            f'<div class="health-gauge-inner"></div>'
            f'<div class="health-gauge-divider"></div>'
            f'<div class="health-gauge-label {gauge_class}">{label}</div>'
            f"</div>"
            f'<p style="text-align:center;font-size:0.8rem;'
            f'color:rgba(255,255,255,0.4);">Severity Level</p>',
            unsafe_allow_html=True,
        )

        if sev in ("HIGH", "CRITICAL"):
            st.markdown(
                '<p style="text-align:center;color:#f44336;font-size:'
                '0.8rem;font-weight:600;">Seek immediate medical '
                "attention!</p>",
                unsafe_allow_html=True,
            )
        elif sev == "MODERATE":
            st.markdown(
                '<p style="text-align:center;color:#FF9800;font-size:'
                '0.8rem;">Seek medical attention if condition '
                "worsens.</p>",
                unsafe_allow_html=True,
            )
        elif sev == "LOW":
            st.markdown(
                '<p style="text-align:center;color:#4CAF50;font-size:'
                '0.8rem;">Monitor symptoms. Rest and stay hydrated.</p>',
                unsafe_allow_html=True,
            )


def _render_quick_actions():
    with st.container(border=False, key="rp_actions"):
        st.markdown(
            '<div class="rp-title">\u26a1 Quick Actions</div>',
            unsafe_allow_html=True,
        )

        if st.button(
            "\U0001F4CD Find Nearby Hospitals",
            key="qa_hospitals",
            use_container_width=True,
        ):
            with st.spinner("Searching nearby hospitals..."):
                hospital_msg = run_find_hospitals()
            _append_msg("assistant", hospital_msg)
            st.rerun()

        if st.button(
            "\U0001FA79 First Aid Guide",
            key="qa_firstaid",
            use_container_width=True,
        ):
            _append_msg("user", "Show me general first-aid tips")
            response = call_conversational_gemini(
                "Provide a brief summary of essential first-aid tips "
                "for common emergencies (bleeding, burns, fractures, "
                "choking). Keep it concise and practical."
            )
            _append_msg("assistant", response)
            st.rerun()

        if st.button(
            "\U0001F4DE Emergency Contacts",
            key="qa_contacts",
            use_container_width=True,
        ):
            _append_msg(
                "assistant",
                "**Emergency Contacts:**\n\n"
                "\U0001F4DE **Rescue 1122** — Dial **1122** "
                "(Ambulance, Fire, Rescue)\n\n"
                "\U0001F4DE **Edhi Foundation** — Dial **115**\n\n"
                "\U0001F3E5 For the nearest hospital, click "
                "**Find Nearby Hospitals** above.",
            )
            st.rerun()

        if st.button(
            "\U0001F4A1 Health Tips",
            key="qa_tips",
            use_container_width=True,
        ):
            _append_msg(
                "assistant",
                "**Quick Health Tips:**\n\n"
                "\U0001FA7A Keep a basic first-aid kit at home "
                "and in your car\n\n"
                "\U0001F4A7 Stay hydrated — aim for 8 glasses "
                "of water daily\n\n"
                "\U0001F3C3 Learn CPR — it can save lives\n\n"
                "\U0001F4F1 Save emergency numbers in your phone",
            )
            st.rerun()


def _render_tips():
    context_tip = _get_contextual_tip()
    with st.container(border=False, key="rp_tips"):
        st.markdown(
        f'<div class="rp-title">\U0001F4A1 Tip</div>'
        f'<div class="tip-card">{context_tip}</div>'
        , unsafe_allow_html=True)


def _get_contextual_tip() -> str:
    sev = st.session_state.severity
    if sev in ("HIGH", "CRITICAL"):
        return (
            "Call <strong>1122</strong> immediately. "
            "Keep the injured person still and calm "
            "until help arrives."
        )
    if sev == "MODERATE":
        return (
            "Clean wounds gently with clean water. "
            "Apply pressure to stop bleeding. "
            "Seek medical care if symptoms persist."
        )
    if st.session_state.is_emergency:
        return (
            "Monitor symptoms closely. "
            "If anything worsens, seek medical attention."
        )
    return (
        "Clean the wound gently with clean water "
        "and apply antiseptic. Cover with a clean bandage."
    )


def _render_analysis_badge():
    result = st.session_state.last_result
    elapsed = st.session_state.last_elapsed or 0
    badge_color = {
        "complete": "#2e7d32",
        "partial": "#f57f17",
    }.get(result.status, "#c62828")

    st.markdown(
        f'<div class="rp-section" style="border-color:{badge_color}40;">'
        f'<div class="rp-title">\U0001F4CA Last Analysis</div>'
        f'<p style="font-size:0.8rem;margin:0.2rem 0;">'
        f'<span style="color:{badge_color};font-weight:600;">'
        f"{result.status.upper()}</span></p>"
        f'<p style="font-size:0.8rem;margin:0;">'
        f"Case: {result.case_type}</p>"
        f'<p style="font-size:0.75rem;color:rgba(255,255,255,0.4);'
        f'margin:0;">{elapsed:.1f}s</p>'
        f"</div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------
# HOSPITAL RESULTS PAGE
# ---------------------------------------------------------


def render_hospitals_page():
    st.markdown(
        '<div style="margin-bottom:1rem;">'
        '<h2 style="color:#fff;margin:0;">'
        "\U0001F3E5 Nearby Hospitals</h2>"
        '<p style="color:rgba(255,255,255,0.5);font-size:0.85rem;'
        'margin:0.3rem 0;">Based on your current location settings</p>'
        "</div>",
        unsafe_allow_html=True,
    )

    with st.spinner("Searching nearby hospitals..."):
        hospital_text = run_find_hospitals()

    st.markdown(
        f'<div class="glass-card">{hospital_text}</div>',
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown(
        "**Location:** "
        f"({st.session_state.latitude:.4f}, "
        f"{st.session_state.longitude:.4f})"
    )
    st.markdown(
        "*Distances are estimates based on static hospital data. "
        "Actual travel times may vary.*"
    )


# ---------------------------------------------------------
# REPORTS PAGE
# ---------------------------------------------------------


def render_reports_page():
    st.markdown(
        '<div style="margin-bottom:1rem;">'
        '<h2 style="color:#fff;margin:0;">'
        "\U0001F4CB Incident Reports</h2>"
        '<p style="color:rgba(255,255,255,0.5);font-size:0.85rem;'
        'margin:0.3rem 0;">Generated from emergency analyses</p>'
        "</div>",
        unsafe_allow_html=True,
    )

    result = st.session_state.get("last_result")
    if not result:
        st.info(
            "No analysis has been run yet. Go to **Assistant** and "
            "describe an emergency to generate a report."
        )
        return

    report = generate_report_text()
    st.markdown(report)

    st.markdown("---")
    st.download_button(
        "\U0001F4E5 Download Report",
        report,
        file_name="guardianai_report.md",
        mime="text/markdown",
    )


# ---------------------------------------------------------
# ABOUT PAGE
# ---------------------------------------------------------


def render_about_page():
    st.markdown(
        """
## GuardianAI — AI Health Guardian

GuardianAI is a multi-agent emergency response system that provides
immediate first-aid guidance when seconds matter.

### How It Works

1. **Chat** with GuardianAI about your health concern
2. **GuardianAI** asks clarifying questions and assesses urgency
3. For emergencies, **four AI agents** work together:
   - **Vision Agent** — analyzes scene photos
   - **Medical Agent** — assesses severity
   - **Knowledge Agent** — retrieves first-aid steps from medical references
   - **Location Agent** — finds the nearest hospital
4. Results merge into a unified assessment with actionable guidance

### Architecture

| Component | Technology |
|---|---|
| Orchestration | LangGraph (StateGraph) |
| AI Models | Google Gemini |
| Knowledge Base | Pinecone (RAG) — medical references |
| Location | Static hospital database (14 cities in Pakistan) |
| Frontend | Streamlit |

### Stats

- **134 tests** passing
- **831 pages** of first-aid knowledge indexed
- **4 specialized agents** + 1 coordinator
- **Graceful degradation** — if any agent fails, others still work

### Built For

Alibaba Cloud Hackathon — AI for Emergency Response
"""
    )


# ---------------------------------------------------------
# CHAT INPUT HANDLER
# ---------------------------------------------------------


def handle_chat_input(user_input: str, uploaded_file):
    """Process user input from the chat area."""
    image = None
    image_bytes = None
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        image_bytes = buf.getvalue()

    _append_msg("user", user_input, image=image_bytes)

    route = route_message(user_input, image is not None)

    if route == "emergency":
        with st.spinner(
            "\U0001F6E1\ufe0f Running emergency analysis..."
        ):
            run_emergency_pipeline(user_input, image)

    elif route == "conversation":
        with st.spinner(
            "\U0001F6E1\ufe0f GuardianAI is thinking..."
        ):
            response = call_conversational_gemini(user_input)
        _append_msg("assistant", response)

    st.rerun()


# ---------------------------------------------------------
# UI: FOOTER
# ---------------------------------------------------------


def render_footer():
    st.markdown(
        '<div class="guardian-footer">'
        '<span class="footer-copy">'
        "\u00a9 2026 GuardianAI. All rights reserved.</span>"
        '<div class="footer-warn">'
        "\u26a0\ufe0f GuardianAI is not a substitute for professional "
        "medical advice. In emergencies, call your local emergency "
        "service.</div>"
        "</div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------


def main():
    load_dotenv()
    init_state()
    st.markdown(CSS, unsafe_allow_html=True)

    # Keep navigation inside the page.  A native Streamlit sidebar can be
    # collapsed by the browser and was leaving an empty, unusable column.
    nav_col, content_col = st.columns([1.05, 4.6], gap="large")
    nav_panel = nav_col.container(border=False, key="left_nav")
    render_sidebar(nav_panel)
    page = st.session_state.nav_page

    with content_col:
        if page == "Assistant":
            render_header()
            if st.session_state.show_right_panel:
                chat_col, right_col = st.columns([3, 1], gap="medium")
            else:
                chat_col = st.container()
                right_col = None

            with chat_col:
                render_chat_area()
                # Nesting chat_input in this column makes it an inline control
                # rather than Streamlit's page-wide, fixed-bottom input.
                submitted = st.chat_input(
                    "Ask GuardianAI about a health concern...",
                    key="main_chat_input",
                    accept_file=True,
                    file_type=["jpg", "jpeg", "png"],
                    width="stretch",
                )
                if submitted is not None:
                    files = submitted.files or []
                    handle_chat_input(
                        submitted.text or "", files[0] if files else None
                    )
            if right_col is not None:
                with right_col:
                    render_right_panel()

        elif page == "Hospitals":
            render_hospitals_page()
            render_footer()
        elif page == "Reports":
            render_reports_page()
            render_footer()
        elif page == "About":
            render_about_page()
            render_footer()
        elif page in ("Profile", "Settings"):
            title = "Your Profile" if page == "Profile" else "Settings"
            st.markdown(f"## {title}")
            st.info("This demo keeps your information only for the current session. "
                    "Location settings are available in the navigation panel.")
            render_footer()


if __name__ == "__main__":
    main()
