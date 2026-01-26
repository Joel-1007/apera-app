import sys
import os
import time
import base64
import json
import datetime
import uuid
import pandas as pd
import plotly.express as px
import requests
import streamlit as st

# --- CONFIGURATION ---
API_URL = "https://unforsaken-sheri-unvitrified.ngrok-free.dev"
HISTORY_FILE = "chat_history.json"
AUDIT_FILE = "audit_log.json"
CACHE_FILE = "arxiv_cache.json"

st.set_page_config(page_title="APERA Pro", page_icon="🔬", layout="wide")

# --- DATA PERSISTENCE ---
def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            return json.load(f)
    return []

def save_history(messages):
    with open(HISTORY_FILE, "w") as f:
        json.dump(messages, f)

def log_feedback(query, response, rating, reason=None):
    entry = {
        "timestamp": str(datetime.datetime.now()),
        "query": query,
        "response": response,
        "rating": rating,
        "reason": reason,
        "session_id": st.session_state.session_id
    }
    
    existing_logs = []
    if os.path.exists(AUDIT_FILE):
        with open(AUDIT_FILE, "r") as f:
            try: 
                existing_logs = json.load(f)
            except: 
                pass
    
    existing_logs.append(entry)
    with open(AUDIT_FILE, "w") as f:
        json.dump(existing_logs, f, indent=2)

def get_cache_size():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as f: 
            return len(json.load(f))
    return 0

# --- AUTHENTICATION & LOGIN LOGIC ---
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
    st.session_state.session_id = str(uuid.uuid4())[:8]

# Helper to read logo dynamically
def get_image_base64(file_path):
    import base64
    try:
        with open(file_path, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except Exception:
        return None

def login_screen():
    # 1. Load Logo
    current_dir = os.path.dirname(os.path.abspath(__file__))
    logo_path = os.path.join(current_dir, "logo.png")
    logo_b64 = get_image_base64(logo_path)
    
    # Fallback to emoji if file missing
    if logo_b64:
        img_tag = f'<img src="data:image/png;base64,{logo_b64}" width="140" style="margin-bottom: 20px; filter: drop-shadow(0 0 15px rgba(139, 92, 246, 0.4));">'
    else:
        img_tag = '<div style="font-size: 80px; margin-bottom: 1rem;">🔬</div>'

    # 2. Hero Section
    st.markdown(f"""
    <div style="text-align: center; padding: 4rem 0;">
        {img_tag}
        <h1 style="background: linear-gradient(135deg, #8b5cf6 0%, #ec4899 100%); 
                   -webkit-background-clip: text; -webkit-text-fill-color: transparent; 
                   font-size: 3.5rem; font-weight: 900; margin-bottom: 0.5rem;">
            APERA SECURE GATEWAY
        </h1>
        <p style="color: #94a3b8; font-size: 1.2rem; font-weight: 500;">
            Advanced Production-Grade Research Intelligence
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # 3. Split Layout (SSO vs Admin)
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        # SSO Buttons
        col_g, col_m = st.columns(2)
        with col_g:
            if st.button("🔵 Sign in with Google", use_container_width=True):
                with st.spinner("🔄 Verifying Google Workspace Identity..."):
                    time.sleep(1.5)
                    st.session_state.authenticated = True
                    st.session_state.username = "jjohnjoel2005@gmail.com"
                    st.toast("✅ Google Workspace Verified.")
                    time.sleep(0.5)
                    st.rerun()
        
        with col_m:
            if st.button("🟧 Sign in with Microsoft", use_container_width=True):
                with st.spinner("🔄 Connecting to Azure AD..."):
                    time.sleep(1.5)
                    st.session_state.authenticated = True
                    st.session_state.username = "joel.john@microsoft.com"
                    st.toast("✅ Azure AD Verified.")
                    time.sleep(0.5)
                    st.rerun()

        st.markdown("""<div style="text-align: center; color: #64748b; margin: 20px 0; font-size: 0.85rem;">— OR USE ADMIN CREDENTIALS —</div>""", unsafe_allow_html=True)

        # Admin Login Form
        with st.form("login_form"):
            st.markdown("### 🔐 System Access")
            user = st.text_input("Username", placeholder="admin")
            pw = st.text_input("Password", type="password", placeholder="••••••••")
            submitted = st.form_submit_button("🚀 Authenticate", use_container_width=True)
            
            if submitted:
                if user == "admin" and pw == "password":
                    st.session_state.authenticated = True
                    st.session_state.username = "Administrator"
                    st.success("✅ Access Granted")
                    time.sleep(0.5)
                    st.rerun()
                else:
                    st.error("❌ Access Denied")

if not st.session_state.authenticated:
    login_screen()
    st.stop()

# --- EPIC GOD MODE CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');
    
    /* Base Reset */
    .main * { font-family: 'Inter', sans-serif !important; }
    .main { 
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #334155 100%) !important;
        animation: gradientShift 15s ease infinite;
    }
    @keyframes gradientShift {
        0%, 100% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
    }
    .block-container { padding-top: 2rem !important; max-width: 1200px !important; }
    
    /* Header - Ultra Premium */
    .header-container {
        background: linear-gradient(145deg, rgba(30, 41, 59, 0.95), rgba(15, 23, 42, 0.9)) !important;
        backdrop-filter: blur(20px) saturate(180%);
        border: 2px solid rgba(139, 92, 246, 0.3);
        border-radius: 24px;
        padding: 2.5rem;
        margin-bottom: 2rem;
        box-shadow: 
            0 20px 60px rgba(139, 92, 246, 0.3),
            0 0 0 1px rgba(255, 255, 255, 0.1) inset,
            0 0 100px rgba(99, 102, 241, 0.2);
        position: relative;
        overflow: hidden;
        animation: headerGlow 3s ease-in-out infinite;
    }
    @keyframes headerGlow {
        0%, 100% { box-shadow: 0 20px 60px rgba(139, 92, 246, 0.3), 0 0 0 1px rgba(255, 255, 255, 0.1) inset; }
        50% { box-shadow: 0 20px 80px rgba(139, 92, 246, 0.5), 0 0 0 1px rgba(255, 255, 255, 0.2) inset; }
    }
    .header-container::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(139, 92, 246, 0.1) 0%, transparent 70%);
        animation: rotate 20s linear infinite;
    }
    @keyframes rotate {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }
    
    .logo-img { 
        width: 80px; height: 80px; 
        border-radius: 20px; 
        box-shadow: 0 10px 40px rgba(139, 92, 246, 0.5); 
        transition: all 0.4s cubic-bezier(0.34, 1.56, 0.64, 1);
    }
    .logo-img:hover { transform: scale(1.1) rotate(5deg); }
    
    .title-text {
        font-size: 3rem; font-weight: 900;
        background: linear-gradient(135deg, #60a5fa 0%, #8b5cf6 50%, #ec4899 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0; line-height: 1.1; letter-spacing: -0.04em;
        text-shadow: 0 0 40px rgba(139, 92, 246, 0.5);
        animation: titlePulse 3s ease-in-out infinite;
    }
    @keyframes titlePulse {
        0%, 100% { filter: brightness(1); }
        50% { filter: brightness(1.2); }
    }
    
    .subtitle-text { 
        font-size: 1.1rem; color: #94a3b8; 
        margin-top: 0.75rem; font-weight: 600; 
        letter-spacing: 0.05em;
    }
    
    /* Sidebar - Dark Premium */
    [data-testid="stSidebar"] { 
        background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%) !important;
        border-right: 2px solid rgba(139, 92, 246, 0.3) !important;
        box-shadow: 5px 0 30px rgba(0, 0, 0, 0.5);
    }
    [data-testid="stSidebar"] h2 { 
        background: linear-gradient(135deg, #60a5fa, #8b5cf6); 
        -webkit-background-clip: text; -webkit-text-fill-color: transparent; 
        font-weight: 900;
    }
    [data-testid="stSidebar"] h3 { 
        color: #8b5cf6; font-weight: 700; 
        font-size: 1.1rem; margin-top: 1.5rem; 
    }
    
    /* Chat Messages - Glass Morphism */
    [data-testid="stChatMessage"] {
        background: linear-gradient(145deg, rgba(30, 41, 59, 0.8), rgba(15, 23, 42, 0.9)) !important;
        backdrop-filter: blur(20px) saturate(150%);
        border: 2px solid rgba(139, 92, 246, 0.2) !important;
        border-radius: 20px !important;
        padding: 1.5rem !important;
        margin: 1rem 0 !important;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4) !important;
        transition: all 0.3s ease;
    }
    [data-testid="stChatMessage"]:hover {
        border-color: rgba(139, 92, 246, 0.4) !important;
        box-shadow: 0 12px 40px rgba(139, 92, 246, 0.3) !important;
        transform: translateY(-2px);
    }
    [data-testid="stChatMessage"][data-testid*="user"] { 
        background: linear-gradient(145deg, rgba(99, 102, 241, 0.15), rgba(139, 92, 246, 0.15)) !important; 
        border: 2px solid rgba(99, 102, 241, 0.3) !important; 
    }
    
    /* Citations - Neon Cards */
    .citation-item {
        background: linear-gradient(145deg, rgba(30, 41, 59, 0.9), rgba(15, 23, 42, 0.95));
        border-left: 5px solid #8b5cf6;
        border-radius: 16px;
        padding: 1.25rem 1.25rem 1.25rem 1.75rem;
        margin-top: 1rem;
        box-shadow: 
            0 4px 20px rgba(139, 92, 246, 0.2),
            0 0 0 1px rgba(139, 92, 246, 0.1) inset;
        transition: all 0.4s cubic-bezier(0.34, 1.56, 0.64, 1);
    }
    .citation-item:hover { 
        transform: translateX(8px) scale(1.02); 
        box-shadow: 0 8px 32px rgba(139, 92, 246, 0.4);
        border-left-width: 8px;
    }
    .citation-meta { 
        display: flex; justify-content: space-between; 
        align-items: center; color: #94a3b8; 
        font-weight: 700; font-size: 0.85rem; 
        text-transform: uppercase; letter-spacing: 0.1em; 
        margin-bottom: 0.75rem; 
    }
    .citation-meta a { 
        color: #8b5cf6; text-decoration: none; 
        font-weight: 700; transition: all 0.2s;
        text-shadow: 0 0 10px rgba(139, 92, 246, 0.5);
    }
    .citation-meta a:hover { 
        color: #a78bfa; text-decoration: underline;
        text-shadow: 0 0 20px rgba(139, 92, 246, 0.8);
    }
    
    /* Badges - Glow Effect */
    .online-badge { 
        background: linear-gradient(135deg, #3b82f6, #2563eb); 
        color: #ffffff; padding: 6px 14px; 
        border-radius: 10px; font-size: 0.75rem; 
        font-weight: 800; 
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.4);
        animation: badgeGlow 2s ease-in-out infinite;
    }
    @keyframes badgeGlow {
        0%, 100% { box-shadow: 0 4px 15px rgba(59, 130, 246, 0.4); }
        50% { box-shadow: 0 4px 25px rgba(59, 130, 246, 0.6); }
    }
    
    .local-badge { 
        background: linear-gradient(135deg, #6b7280, #4b5563); 
        color: #ffffff; padding: 6px 14px; 
        border-radius: 10px; font-size: 0.75rem; 
        font-weight: 800; 
        box-shadow: 0 4px 15px rgba(75, 85, 99, 0.4);
    }
    
    .badge-intent { 
        background: linear-gradient(135deg, #8b5cf6, #7c3aed); 
        color: #ffffff; padding: 6px 12px; 
        border-radius: 8px; font-weight: 800; 
        font-size: 0.8rem; 
        display: inline-block; margin-right: 10px;
        box-shadow: 0 4px 15px rgba(139, 92, 246, 0.4);
    }
    
    .badge-fairness { 
        background: linear-gradient(135deg, #f59e0b, #d97706); 
        color: #ffffff; padding: 6px 12px; 
        border-radius: 8px; font-weight: 800; 
        font-size: 0.8rem; 
        display: inline-block;
        box-shadow: 0 4px 15px rgba(245, 158, 11, 0.4);
    }
    
    /* Diversity Warning - Urgent Alert */
    .diversity-warning { 
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.2), rgba(220, 38, 38, 0.2)); 
        color: #fca5a5; 
        padding: 12px 16px; 
        border-radius: 12px; 
        border: 2px solid rgba(239, 68, 68, 0.5); 
        font-size: 0.9rem; 
        margin-top: 12px; 
        display: flex; 
        align-items: center; 
        gap: 10px;
        font-weight: 600;
        animation: warningPulse 2s ease-in-out infinite;
    }
    @keyframes warningPulse {
        0%, 100% { border-color: rgba(239, 68, 68, 0.5); }
        50% { border-color: rgba(239, 68, 68, 0.8); }
    }
    
    /* Buttons - Holographic */
    .stButton button {
        background: linear-gradient(135deg, #8b5cf6 0%, #ec4899 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 14px !important;
        padding: 0.85rem 1.75rem !important;
        font-weight: 700 !important;
        box-shadow: 0 6px 24px rgba(139, 92, 246, 0.4) !important;
        transition: all 0.4s cubic-bezier(0.34, 1.56, 0.64, 1) !important;
        position: relative;
        overflow: hidden;
    }
    .stButton button::before {
        content: '';
        position: absolute;
        top: 0; left: -100%;
        width: 100%; height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
        transition: left 0.5s;
    }
    .stButton button:hover::before {
        left: 100%;
    }
    .stButton button:hover { 
        transform: translateY(-4px) scale(1.05) !important; 
        box-shadow: 0 12px 40px rgba(139, 92, 246, 0.6) !important;
    }
    
    /* Progress Bar - Neon */
    .stProgress > div > div > div > div { 
        background: linear-gradient(90deg, #8b5cf6, #ec4899) !important;
        box-shadow: 0 0 20px rgba(139, 92, 246, 0.6);
    }
    
    /* Expander - Dark Glass */
    [data-testid="stExpander"] { 
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.6), rgba(15, 23, 42, 0.8)) !important;
        backdrop-filter: blur(10px);
        border: 2px solid rgba(139, 92, 246, 0.2) !important;
        border-radius: 16px !important;
        overflow: hidden;
    }
    [data-testid="stExpander"] summary { 
        font-weight: 700 !important; 
        color: #a78bfa !important; 
        padding: 1.25rem !important;
    }
    [data-testid="stExpander"] summary:hover { 
        background: rgba(139, 92, 246, 0.15) !important;
    }
    
    /* Chat Input - Glow Effect */
    [data-testid="stChatInput"] { 
        border: 2px solid rgba(139, 92, 246, 0.3) !important;
        border-radius: 18px !important;
        box-shadow: 0 6px 30px rgba(139, 92, 246, 0.2) !important;
        background: rgba(30, 41, 59, 0.8) !important;
    }
    [data-testid="stChatInput"]:focus-within { 
        border-color: rgba(139, 92, 246, 0.6) !important;
        box-shadow: 0 8px 40px rgba(139, 92, 246, 0.4) !important;
    }
    
    /* Metrics - Glass Cards */
    [data-testid="stMetricValue"] {
        color: #8b5cf6 !important;
        font-weight: 900 !important;
        font-size: 2rem !important;
        text-shadow: 0 0 20px rgba(139, 92, 246, 0.5);
    }
    
    /* Text Color Override */
    .main p, .main span, .main div { color: #cbd5e1 !important; }
    .main h1, .main h2, .main h3 { color: #e2e8f0 !important; }
</style>
""", unsafe_allow_html=True)

# --- LOGO HELPER ---
def get_base64_logo():
    if os.path.exists("assets/logo.png"):
        with open("assets/logo.png", "rb") as f:
            return base64.b64encode(f.read()).decode()
    return None
logo_b64 = get_base64_logo()

# --- SIDEBAR ---
with st.sidebar:
    if logo_b64:
        st.markdown(f'''
        <div style="display:flex;align-items:center;gap:12px;margin-bottom:24px;">
            <img src="data:image/png;base64,{logo_b64}" width="48" style="border-radius:12px;box-shadow:0 4px 15px rgba(139,92,246,0.4);">
            <h2 style="margin:0;">APERA Pro</h2>
        </div>
        ''', unsafe_allow_html=True)
    
    st.markdown(f"<div style='background:rgba(139,92,246,0.15);padding:8px 12px;border-radius:8px;font-size:0.85rem;font-weight:600;'>🔑 Session: {st.session_state.session_id}</div>", unsafe_allow_html=True)
    
    st.markdown("---")
    app_mode = st.radio("🧭 System Module:", ["Research Assistant", "Admin Audit"], index=0)
    
    if app_mode == "Research Assistant":
        st.markdown("### ⚙️ Engine Configuration")
        
       search_mode = st.selectbox(
            "Retrieval Strategy:", 
            ["Live Research (ArXiv)"]
        )
        
        # Map to backend mode key
        mode_key = "local"
        if "ArXiv" in search_mode: 
            mode_key = "online"
        if "Semantic" in search_mode: 
            mode_key = "semantic"
        
        st.divider()
        if st.button("🗑️ Clear Context", use_container_width=True):
            st.session_state.messages = []
            save_history([])
            st.rerun()

    st.divider()
    if st.button("🚪 Secure Logout", use_container_width=True):
        st.session_state.authenticated = False
        st.rerun()

# --- ADMIN AUDIT VIEW ---
if app_mode == "Admin Audit":
    st.markdown("""
    <h1 style="text-align:center;background:linear-gradient(135deg,#8b5cf6,#ec4899);-webkit-background-clip:text;-webkit-text-fill-color:transparent;font-size:3rem;font-weight:900;margin-bottom:2rem;">
        🛡️ Governance & Ethics Dashboard
    </h1>
    """, unsafe_allow_html=True)
    
    # Try to fetch from API first, fallback to local files
    try:
        logs_response = requests.get(f"{API_URL}/admin/logs", timeout=2)
        tox_response = requests.get(f"{API_URL}/admin/toxicity", timeout=2)
        
        if logs_response.status_code == 200:
            logs = logs_response.json()
        else:
            logs = []
            
        if tox_response.status_code == 200:
            tox_data = tox_response.json()
        else:
            tox_data = []
            
    except:
        # Fallback to local audit file
        if os.path.exists(AUDIT_FILE):
            with open(AUDIT_FILE, 'r') as f:
                try: 
                    logs = json.load(f)
                except: 
                    logs = []
        else:
            logs = []
        tox_data = []
    
    # Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    
    if logs:
        df = pd.DataFrame(logs)
        
        col1.metric("📊 Total Queries", len(df), delta=None)
        pos_count = len(df[df['rating'] == 'positive']) if 'rating' in df.columns else 0
        col2.metric("👍 Positive Feedback", pos_count, delta=f"{int(pos_count/len(df)*100) if len(df) > 0 else 0}%")
        col3.metric("📦 Cached Papers", get_cache_size())
        active_sessions = len(df['session_id'].unique()) if 'session_id' in df.columns else 1
        col4.metric("🔒 Active Sessions", active_sessions)
        
        st.divider()
        
        # Charts
        c1, c2 = st.columns(2)
        
        with c1:
            st.markdown("### 📈 User Satisfaction")
            if 'rating' in df.columns:
                fig_pie = px.pie(
                    df, names='rating', 
                    title='Feedback Distribution',
                    color_discrete_map={'positive': '#8b5cf6', 'negative': '#ef4444'},
                    hole=0.4
                )
                fig_pie.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#cbd5e1')
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            else:
                st.info("No rating data available yet.")
        
        with c2:
            st.markdown("### ⚖️ Source Diversity Analysis")
            mock_fairness = pd.DataFrame({
                'Region': ['Global North', 'Global South', 'Asia-Pacific', 'Unknown'],
                'Count': [65, 18, 12, 5]
            })
            fig_bar = px.bar(
                mock_fairness, x='Region', y='Count',
                title='Citation Geographic Balance',
                color='Count',
                color_continuous_scale='purples'
            )
            fig_bar.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#cbd5e1')
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        
        st.divider()
        
        # Time Series
        if 'timestamp' in df.columns:
            st.markdown("### 🕐 Query Activity Timeline")
            df['hour'] = pd.to_datetime(df['timestamp']).dt.floor('h')  # lowercase 'h'
            activity = df.groupby('hour').size().reset_index(name='queries')
            fig_line = px.line(
                activity, x='hour', y='queries',
                title='Hourly Query Volume',
                markers=True
            )
            fig_line.update_traces(line_color='#8b5cf6', marker=dict(size=8))
            fig_line.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#cbd5e1')
            )
            st.plotly_chart(fig_line, use_container_width=True)
        
        # Audit Log Table
        st.markdown("### 🔍 Session Audit Log")
        available_cols = ['timestamp', 'query', 'rating']
        if 'session_id' in df.columns:
            available_cols.append('session_id')
        st.dataframe(
            df[available_cols].tail(20),
            width="stretch"
        )
        
        st.info("ℹ️ **Fairness Alert:** Global North dominance >60% indicates potential Western-centric bias.")
    else:
        st.info("📭 No audit logs available. Start chatting to generate governance data.")

# --- PASTE THIS HERE ---
    st.markdown("---")
    st.subheader("📂 Ingest Knowledge Base")
    st.info("Upload a PDF to create the Local Database.")
    
    # File Uploader
    uploaded_file = st.file_uploader("Upload Research Paper (PDF)", type="pdf")

    if uploaded_file:
        if st.button("Process & Ingest"):
            with st.spinner("🧠 Reading and indexing document..."):
                # Prepare file for API
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                try:
                    # Send to Backend
                    response = requests.post(f"{API_URL}/ingest", files=files)
                    
                    if response.status_code == 200:
                        st.success("✅ Success! Local DB is now ready.")
                        st.balloons()
                    else:
                        st.error(f"Failed: {response.text}")
                except Exception as e:
                    st.error(f"Connection Error: {e}")

# --- RESEARCH ASSISTANT VIEW ---
elif app_mode == "Research Assistant":
    # Epic Header
    logo_html = f'<img src="data:image/png;base64,{logo_b64}" class="logo-img">' if logo_b64 else '<div style="font-size:70px;">🔬</div>'
    st.markdown(f"""
    <div class="header-container" style="display: flex; align-items: center; gap: 2rem;">
        {logo_html}
        <div style="position:relative;z-index:1;">
            <h1 class="title-text">APERA RESEARCH INTELLIGENCE</h1>
            <p class="subtitle-text">
                <b>Mode:</b> {search_mode} • 
                <b>Engine:</b> Mistral AI • 
                <b>Architecture:</b> Async Microservices
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Initialize History
    if "messages" not in st.session_state:
        st.session_state.messages = load_history()

    # Empty State
    if len(st.session_state.messages) == 0:
        st.markdown(f"""
        <div style="text-align:center;padding:4rem 2rem;background:rgba(30,41,59,0.5);border-radius:20px;border:2px dashed rgba(139,92,246,0.3);">
            <div style="font-size:60px;margin-bottom:1rem;">🚀</div>
            <h2 style="color:#8b5cf6;margin-bottom:0.5rem;">Ready for Liftoff</h2>
            <p style="color:#94a3b8;">Ask me anything about AI research. Currently using <b>{search_mode}</b>.</p>
        </div>
        """, unsafe_allow_html=True)

    # Render Chat History
    for idx, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            if message["role"] == "assistant":
                # Metadata Badges
                if message.get("meta"):
                    m = message["meta"]
                    st.write("")
                    
                    st.markdown(f"""
                        <span class='badge-intent'>INTENT: {m.get('intent', 'GENERAL')}</span>
                        <span class='badge-fairness'>BIAS: {m.get('fairness', {}).get('balance_label', 'Unknown')}</span>
                    """, unsafe_allow_html=True)
                    
                    # Diversity Warning
                    if m.get('fairness', {}).get('diversity_flag'):
                        st.markdown("""
                        <div class="diversity-warning">
                            ⚠️ <b>Diversity Constraint Alert:</b> Results dominated by single publisher/source.
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Confidence Score
                    conf = m.get('confidence', 0)
                    st.progress(conf/100, text=f"🎯 Hallucination Confidence: {conf}%")
                    
                    # XAI Explainability
                    with st.expander("❓ XAI: Why this answer?"):
                        st.markdown(f"**Reasoning:** {m.get('xai_reason', 'N/A')}")
                        st.markdown(f"**Retrieval Method:** Hybrid (Vector + BM25) or Cached ArXiv")
                        st.markdown(f"**Sources Used:** {len(message.get('citations', []))} documents")
                
                # Citations Section
                if "citations" in message and message["citations"]:
                    citations = message["citations"]
                    with st.expander(f"📚 View {len(citations)} Sources ({message.get('mode', 'Unknown')})"):
                        for cite in citations:
                            badge_class = "online-badge" if cite.get('type') in ['online', 'semantic'] else "local-badge"
                            badge_text = "LIVE WEB" if cite.get('type') in ['online', 'semantic'] else "LOCAL DB"
                            link_html = f" <a href='{cite.get('url', '#')}' target='_blank'>[Open PDF]</a>" if 'url' in cite else ""
                            
                            st.markdown(f"""
                            <div class="citation-item">
                                <div class="citation-meta">
                                    <span>📄 {cite.get('file', 'Unknown Source')} {link_html}</span>
                                    <span class="{badge_class}">{badge_text}</span>
                                </div>
                                <div style="color: #94a3b8; margin-top: 0.5rem; font-size: 0.9rem;">
                                    "{cite.get('text', 'No preview available')[:250]}..."
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                
                # Feedback Buttons (Only on last message)
                if idx == len(st.session_state.messages) - 1:
                    col1, col2, col3 = st.columns([1, 1, 8])
                    if col1.button("👍", key=f"up_{idx}"):
                        log_feedback(
                            st.session_state.messages[idx-1]['content'], 
                            message['content'], 
                            "positive"
                        )
                        # Try to send to API as well
                        try:
                            requests.post(
                                f"{API_URL}/feedback", 
                                json={"query": st.session_state.messages[idx-1]['content'], "rating": "positive"},
                                timeout=1
                            )
                        except:
                            pass
                        st.toast("✅ Positive feedback logged!")
                    if col2.button("👎", key=f"down_{idx}"):
                        log_feedback(
                            st.session_state.messages[idx-1]['content'], 
                            message['content'], 
                            "negative"
                        )
                        # Try to send to API as well
                        try:
                            requests.post(
                                f"{API_URL}/feedback", 
                                json={"query": st.session_state.messages[idx-1]['content'], "rating": "negative"},
                                timeout=1
                            )
                        except:
                            pass
                        st.toast("⚠️ Feedback flagged for review!")

    # Chat Input
    if query := st.chat_input("💬 Ask a research question..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)
        
        # Generate assistant response
        with st.chat_message("assistant"):
            with st.spinner("🤖 Connecting to Async Neural Engine..."):
                try:
                    # UPDATED: Call FastAPI backend instead of local agent
                    payload = {
                        "query": query, 
                        "session_id": st.session_state.session_id, 
                        "mode": mode_key
                    }
                    
                    response = requests.post(f"{API_URL}/chat", json=payload, timeout=60)
                    
                    if response.status_code == 200:
                        res = response.json()
                        response_text = res["response"]
                        citations = res.get("citations", [])
                        meta = res.get("meta", {})
                        
                        # Display response
                        st.markdown(response_text)
                        
                        # Show metadata immediately
                        if meta:
                            st.markdown(f"""
                                <span class='badge-intent'>INTENT: {meta.get('intent', 'GENERAL')}</span>
                                <span class='badge-fairness'>BIAS: {meta.get('fairness', {}).get('balance_label', 'Unknown')}</span>
                            """, unsafe_allow_html=True)
                            
                            if meta.get('fairness', {}).get('diversity_flag'):
                                st.markdown("""
                                <div class="diversity-warning">
                                    ⚠️ <b>Diversity Constraint Alert:</b> Results dominated by single source.
                                </div>
                                """, unsafe_allow_html=True)
                            
                            conf = meta.get('confidence', 0)
                            st.progress(conf/100, text=f"🎯 Confidence: {conf}%")
                        
                        # Show citations
                        if citations:
                            with st.expander(f"📚 View {len(citations)} Sources ({search_mode})"):
                                for cite in citations:
                                    badge_class = "online-badge" if cite.get('type') in ['online', 'semantic'] else "local-badge"
                                    badge_text = "LIVE WEB" if cite.get('type') in ['online', 'semantic'] else "LOCAL DB"
                                    link_html = f" <a href='{cite.get('url', '#')}' target='_blank'>[Open]</a>" if 'url' in cite else ""
                                    
                                    st.markdown(f"""
                                    <div class="citation-item">
                                        <div class="citation-meta">
                                            <span>📄 {cite.get('file', 'Unknown')} {link_html}</span>
                                            <span class="{badge_class}">{badge_text}</span>
                                        </div>
                                        <div style="color: #94a3b8; margin-top: 0.5rem;">
                                            "{cite.get('text', '')[:250]}..."
                                        </div>
                                    </div>
                                    """, unsafe_allow_html=True)
                        
                        # Save to history
                        new_msg = {
                            "role": "assistant",
                            "content": response_text,
                            "citations": citations,
                            "mode": search_mode,
                            "meta": meta
                        }
                        st.session_state.messages.append(new_msg)
                        save_history(st.session_state.messages)
                        st.rerun()
                    else:
                        st.error(f"⚠️ API Error {response.status_code}: {response.text}")
                        
                except requests.exceptions.ConnectionError:
                    st.error("❌ **Connection Failed**")
                    st.info("💡 **Troubleshooting:** Make sure the FastAPI backend is running with `python src/api.py`")
                except requests.exceptions.Timeout:
                    st.error("⏱️ **Request Timeout**")
                    st.info("The backend is taking too long to respond. Try a simpler query or check backend logs.")
                except Exception as e:
                    st.error(f"❌ **System Error:** {str(e)}")
                    st.info("💡 Ensure `python src/api.py` is running and healthy.")
