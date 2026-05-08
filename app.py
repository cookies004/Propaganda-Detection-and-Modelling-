import streamlit as st
import numpy as np 
import networkx as nx
import random
import time
import pandas as pd
import joblib

st.set_page_config(layout="wide", page_title="Propaganda Propagation Simulator")

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_models():
    model = joblib.load("propaganda_logistic_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_models()

# =========================
# CUSTOM CSS
# =========================
st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .stApp { background-color: #0f1117; }
    .sim-header {
        font-size: 1.6rem;
        font-weight: 700;
        color: #f0f0f0;
        margin-bottom: 0.2rem;
        letter-spacing: -0.5px;
    }
    .sim-sub {
        font-size: 0.85rem;
        color: #888;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #1a1d27;
        border: 1px solid #2a2d3a;
        border-radius: 10px;
        padding: 14px 18px;
        text-align: center;
    }
    .metric-label {
        font-size: 0.7rem;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
    }
    .badge-prop {
        display: inline-block;
        background: #3d1515;
        color: #ff6b6b;
        border: 1px solid #ff6b6b44;
        border-radius: 20px;
        padding: 4px 14px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .badge-safe {
        display: inline-block;
        background: #0f2d1a;
        color: #51cf66;
        border: 1px solid #51cf6644;
        border-radius: 20px;
        padding: 4px 14px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .panel-title {
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        padding: 8px 14px;
        border-radius: 8px 8px 0 0;
        margin-bottom: 4px;
    }
    .prop-panel { background: #2d1515; color: #ff6b6b; }
    .mit-panel  { background: #0f2d1a; color: #51cf66; }
    .stButton > button {
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
    }
</style>
""", unsafe_allow_html=True)

# =========================
# SESSION STATE
# =========================
defaults = {
    "label": None,
    "prob": 0.7,
    "sim_running": False,
    "G": None,
    "status_prop": None,
    "status_mit": None,
    "history_prop": [],
    "history_mit": [],
    "step": 0,
    "positions": None,
    "blocked": set(),
    "beta": 0.7,
    "propagation": "Uniform",
    "mitigation": "Block Hubs",
    "max_steps": 50,
    "use_manual_beta": False,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# =========================
# GRAPH BUILDER
# =========================
def build_graph(n, m=2):
    return nx.barabasi_albert_graph(n, m, seed=42)

def compute_blocked(G, mitigation, block_percent=0.2):
    degs = dict(G.degree())
    sorted_nodes = sorted(degs, key=degs.get, reverse=True)

    k = int(block_percent * len(G))

    if mitigation == "Block Hubs":
        return set(sorted_nodes[:k])

    elif mitigation == "Block Bridges":
        return set(sorted_nodes[-k:])

    return set()

# =========================
# SI STEP
# status: 0=susceptible, 1=infected, 2=immune/blocked
# =========================
def si_step(G, status, beta, propagation, blocked, mitigation):
    degs = dict(G.degree())
    vals = list(degs.values())
    mean_deg = np.mean(vals)
    median_deg = np.median(vals)
    new_status = status.copy()
    for node in G.nodes():
        if status[node] != 1:
            continue

        for nei in G.neighbors(node):

            if status[nei] != 0:
                continue

            spread_ok = True

            if propagation == "Hub":
                spread_ok = degs[nei] > mean_deg
            elif propagation == "Bridge":
                spread_ok = degs[nei] < median_deg

            effective_beta = beta
            if mitigation == "Block Hubs" and node in blocked:
                effective_beta = beta * 0.2

            elif mitigation == "Block Bridges" and node in blocked:
                effective_beta = beta * 0.4

            elif mitigation == "Hybrid" and node in blocked:
                effective_beta = beta * 0.1

            if spread_ok and random.random() < effective_beta:
                new_status[nei] = 1
    return new_status

# =========================
# GRAPH -> SVG
# =========================
def graph_to_svg(G, status, positions, start_node=0, width=420, height=300):
    nodes = list(G.nodes())
    if not positions:
        return "<svg></svg>"

    xs = [positions[n][0] for n in nodes]
    ys = [positions[n][1] for n in nodes]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    pad = 24

    def tx(x):
        if xmax == xmin:
            return width / 2
        return pad + (x - xmin) / (xmax - xmin) * (width - 2 * pad)

    def ty(y):
        if ymax == ymin:
            return height / 2
        return pad + (y - ymin) / (ymax - ymin) * (height - 2 * pad)

    n = len(nodes)
    r = max(3, min(7, 280 / n))

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'style="background:#12151e;border-radius:0 0 10px 10px;">'
    ]

    for u, v in G.edges():
        su, sv = status.get(u, 0), status.get(v, 0)
        inf_both = su == 1 and sv == 1
        color = "#ff6b6b44" if inf_both else "#ffffff0d"
        lw = 1.2 if inf_both else 0.6
        parts.append(
            f'<line x1="{tx(positions[u][0]):.1f}" y1="{ty(positions[u][1]):.1f}" '
            f'x2="{tx(positions[v][0]):.1f}" y2="{ty(positions[v][1]):.1f}" '
            f'stroke="{color}" stroke-width="{lw}"/>'
        )

    for nd in nodes:
        x, y = tx(positions[nd][0]), ty(positions[nd][1])
        s = status.get(nd, 0)
        is_start = nd == start_node

        if is_start:
            fill = "#378ADD"
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r+5}" fill="#378ADD22"/>')
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r+2.5}" fill="none" stroke="#378ADD" stroke-width="1.5" opacity="0.5"/>')
        elif s == 1:
            fill = "#E24B4A"
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r+3.5}" fill="#E24B4A18"/>')
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r+1.5}" fill="none" stroke="#E24B4A" stroke-width="1" opacity="0.4"/>')
        elif s == 2:
            fill = "#BA7517"
        else:
            fill = "#639922"

        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{fill}"/>')

    parts.append("</svg>")
    return "".join(parts)

# =========================
# HEADER
# =========================
st.markdown('<div class="sim-header">Propaganda Propagation & Mitigation Simulator</div>', unsafe_allow_html=True)
st.markdown('<div class="sim-sub">Dual-panel SI model — compare raw spread vs. active mitigation in real time</div>', unsafe_allow_html=True)

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.markdown("### Controls")

    nodes_count = st.slider("Network size", 30, 120, 60)
    speed = st.slider("Speed (steps/sec)", 1, 10, 5)
    st.session_state.max_steps = st.slider(
        "Max_steps",
        10,
        200,
        st.session_state.max_steps
    )

    propagation = st.selectbox(
        "Propagation type",
        ["Uniform", "Hub", "Bridge"],
        index=0
    )
    mitigation = st.selectbox(
        "Mitigation strategy",
        ["None", "Block Hubs", "Block Bridges", "Penalize Hubs", "Hybrid"],
        index=1  # Block Hubs by default
    )
        # 🔍 Explanation of selected mitigation
    if mitigation == "None":
        st.info("No mitigation applied. Propagation will spread freely.")
        
    elif mitigation == "Block Hubs":
        st.warning(" Blocking Hubs: Highly connected nodes (influencers) have reduced spread power.")

    elif mitigation == "Block Bridges":
        st.warning(" Blocking Bridges: Nodes connecting communities have reduced influence.")
        
    elif mitigation == "Penalize Hubs":
        st.warning(" Penalizing Hubs: Influential nodes can still spread, but with reduced probability.")
        
    elif mitigation == "Hybrid":
        st.success(" Hybrid Strategy: Both hubs and bridges are controlled for maximum mitigation.")

    
    start_node = st.number_input("Start node", min_value=0, max_value=nodes_count - 1, value=0)

    st.session_state.propagation = propagation
    st.session_state.mitigation = mitigation
    

    st.markdown("---")
    st.markdown("**Legend**")
    st.markdown("🔵 Start node")
    st.markdown("🔴 Infected")
    st.markdown("🟢 Susceptible")
    st.markdown("🟡 Blocked / immune")

# =========================
# TEXT ANALYSIS
# =========================
with st.expander("📝 Content Analysis", expanded=True):
    text_input = st.text_area("Paste content to analyze", height=80,
                               placeholder="Enter text to score propaganda probability...")
    col_btn, col_prob = st.columns([1, 3])
    with col_btn:
        analyze = st.button("Analyze", type="primary", use_container_width=True)
    with col_prob:
        if st.session_state.label:
            prob_pct = int(st.session_state.prob * 100)
            badge_cls = "badge-prop" if st.session_state.label == "Propaganda" else "badge-safe"
            st.markdown(
                f'<div class="{badge_cls}">{st.session_state.label} — {prob_pct}% probability</div>',
                unsafe_allow_html=True
            )
            st.progress(st.session_state.prob)

    if analyze and text_input.strip():
        try:
            
            vec = vectorizer.transform([text_input])
            pred = model.predict(vec)[0]
            prob = float(model.predict_proba(vec)[0][1])
            st.session_state.label = "Propaganda" if pred == 1 else "Non-Propaganda"
            st.session_state.prob = prob
            if pred == 1:
                st.session_state.beta = prob
                st.session_state.max_steps = 50
            else:
                st.session_state.beta = 0
                st.session_state.max_steps = 0
                st.session_state.sim_running = False
                st.session_state.G = None
                st.session_state.status_prop = None
                st.session_state.status_mit = None
                st.session_state.history_prop = []
                st.session_state.history_mit = []
                st.session_state.step = 0    

            st.session_state.use_manual_beta = False
            st.rerun()
        except Exception as e:
            st.error(f"Analysis error: {e}")
    if st.session_state.label is not None:
        st.markdown("### How analysis works")
        st.info("Step 1: Your text is converted into numerical form using TF-IDF vectorization.")
        st.info("Step 2: The trained Logistic Regression model analyzes patterns in the text.")
        st.info(f"Step 3: The model predicts **{st.session_state.label}** with probability **{round(st.session_state.prob, 2)}**.")        

    manual_beta = st.slider(
        "Manual β (transmission rate)", 0.0, 1.0,
        float(st.session_state.prob), 0.01,
        help="Override: set transmission rate directly"
    )
    if manual_beta != st.session_state.beta:
        st.session_state.beta = manual_beta
        st.session_state.use_manual_beta = True

# =========================
# SIM BUTTONS
# =========================
b1, b2, b3, _ = st.columns([1, 1, 1, 3])
with b1:
    start_btn = st.button("▶ Start", type="primary", use_container_width=True)
with b2:
    stop_btn = st.button("■ Stop", use_container_width=True)
with b3:
    reset_btn = st.button("↺ Reset", use_container_width=True)

if start_btn:
    if st.session_state.label != "Propaganda":
        st.warning("Non-propaganda content detected. Simulation is only available for propaganda content.")
    else:
        G = build_graph(nodes_count)
        pos = nx.spring_layout(G, seed=42, k=1.8)
        blocked = compute_blocked(G, mitigation)

        # Propagation: everyone susceptible except start node
        status_prop = {n: 0 for n in G.nodes()}
        status_prop[start_node] = 1

        # Mitigation: blocked nodes start as immune (2) — cannot ever be infected
        status_mit = {n: 0 for n in G.nodes()}
        status_mit[start_node] = 1
        for b in blocked:
            if b != start_node:
                status_mit[b] = 2

        st.session_state.G = G
        st.session_state.positions = pos
        st.session_state.blocked = blocked
        st.session_state.status_prop = status_prop
        st.session_state.status_mit = status_mit
        st.session_state.history_prop = [1]
        st.session_state.history_mit = [1]
        st.session_state.step = 0
        st.session_state.sim_running = True


        if len(blocked) == 0:
            st.warning("⚠️ No nodes blocked — both panels identical. Pick a mitigation strategy.")
        else:
            st.success(f"✅ {len(blocked)} nodes immunized by '{mitigation}'.")

if stop_btn:
    st.session_state.sim_running = False

if reset_btn:
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

# =========================
# METRICS ROW
# =========================
G = st.session_state.G
inf_p = sum(1 for v in (st.session_state.status_prop or {}).values() if v == 1)
inf_m = sum(1 for v in (st.session_state.status_mit  or {}).values() if v == 1)
saved = max(0, inf_p - inf_m)

def metric_card(label, value, color):
    return f'''<div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value" style="color:{color}">{value}</div>
    </div>'''

m1, m2, m3, m4 = st.columns(4)
with m1:
    st.markdown(metric_card("Propagation infected", inf_p, "#ff6b6b"), unsafe_allow_html=True)
with m2:
    st.markdown(metric_card("Mitigation infected", inf_m, "#51cf66"), unsafe_allow_html=True)
with m3:
    st.markdown(metric_card("Step", st.session_state.step, "#ffd43b"), unsafe_allow_html=True)
with m4:
    st.markdown(metric_card("Saved by mitigation", saved, "#74c0fc"), unsafe_allow_html=True)

st.markdown("<div style='margin:10px 0'></div>", unsafe_allow_html=True)

# =========================
# DUAL PANEL GRAPHS
# =========================
col_prop, col_mit = st.columns(2)
svg_placeholder_prop = col_prop.empty()
svg_placeholder_mit  = col_mit.empty()



def render_panels():
    if not st.session_state.G:
        return
    svg_prop = graph_to_svg(
        st.session_state.G,
        st.session_state.status_prop or {},
        st.session_state.positions,
        start_node
    )
    svg_mit = graph_to_svg(
        st.session_state.G,
        st.session_state.status_mit or {},
        st.session_state.positions,
        start_node
    )
    with col_prop:
        st.markdown('<div class="panel-title prop-panel">🔴 Propagation only</div>', unsafe_allow_html=True)
        svg_placeholder_prop.markdown(svg_prop, unsafe_allow_html=True)
    with col_mit:
        st.markdown('<div class="panel-title mit-panel">🟢 With mitigation</div>', unsafe_allow_html=True)
        svg_placeholder_mit.markdown(svg_mit, unsafe_allow_html=True)

def render_chart():
    hp = st.session_state.history_prop
    hm = st.session_state.history_mit
    if not hp:
        return
    df = pd.DataFrame({
        "Step": list(range(len(hp))),
        "Propagation (no mitigation)": hp,
        "With mitigation": hm,
    }).set_index("Step")
    st.line_chart(df, color=["#E24B4A", "#639922"])



render_panels()
st.markdown("###  What is happening right now")
if st.session_state.sim_running:
    st.warning(
        f"Step {st.session_state.step}: "
        "Infected nodes are spreading propaganda to their neighbors."
    )

# Mitigation explanation
    if st.session_state.mitigation == "Block Hubs":
        st.info("High-degree nodes (influencers) are blocked → slows rapid spread.")

    elif st.session_state.mitigation == "Block Bridges":
        st.info("Bridge nodes are blocked → prevents spread between groups.")

    elif st.session_state.mitigation == "Hybrid":
        st.info("Both hubs and bridges are blocked → strongest control applied.")

elif st.session_state.label == "Non-Propaganda":
    st.success("Safe content detected. No propaganda propagation simulation required.")

elif st.session_state.step > 0:
    st.success("Simulation finished. Final spread displayed.")


render_chart()

st.markdown("###  What does this mean?")

st.markdown(
    '<div style="color:#E24B4A; font-weight:600;">🔴 Red line = Propagation without mitigation</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div style="color:#639922; font-weight:600;">🟢 Green line = With mitigation applied</div>',
    unsafe_allow_html=True
)    

st.success("If green is lower than red → mitigation is working ")
st.warning("If both lines are similar → mitigation is weak ")

 
    

# =========================
# SIMULATION LOOP
# =========================
if st.session_state.sim_running and st.session_state.G:
    delay = max(0.05, 1.0 / speed)
    blocked = compute_blocked(st.session_state.G, st.session_state.mitigation)
    st.session_state.status_prop = si_step(
        st.session_state.G,
        st.session_state.status_prop,
        st.session_state.beta,
        st.session_state.propagation,
        blocked,
        st.session_state.mitigation
    )
    st.session_state.status_mit = si_step(
        st.session_state.G,
        st.session_state.status_mit,
        st.session_state.beta,
        st.session_state.propagation,
        blocked,
        st.session_state.mitigation
    )

    inf_p_new = sum(1 for v in st.session_state.status_prop.values() if v == 1)
    inf_m_new = sum(1 for v in st.session_state.status_mit.values()  if v == 1)
#===============================
# SIMULATION EXPLANATION (NEW)
# ===============================

    

    st.session_state.history_prop.append(inf_p_new)
    st.session_state.history_mit.append(inf_m_new)
    st.session_state.step += 1

    susceptible_prop = sum(1 for v in st.session_state.status_prop.values() if v == 0)
    susceptible_mit  = sum(1 for v in st.session_state.status_mit.values()  if v == 0)

    if (st.session_state.step >= st.session_state.max_steps
            or (susceptible_prop == 0 and susceptible_mit == 0)):
        st.session_state.sim_running = False
        st.success(
            f"✅ Done at step {st.session_state.step} — "
            f"Propagation: {inf_p_new} infected | "
            f"Mitigation: {inf_m_new} infected | "
            f"Saved: {max(0, inf_p_new - inf_m_new)}"
        )

    time.sleep(delay)
    if st.session_state.sim_running:
        st.rerun()
