"""
EuroSAT Crop Type Classification — Home Page
Author : Charith Manujaya
Run    : streamlit run app.py
"""

import streamlit as st

st.set_page_config(
    page_title="EuroSAT · Crop Classification",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
html, body, [class*="css"], .stButton button, .stTextInput input,
.stNumberInput input, .stDataFrame, code, pre {
    font-family: 'Courier New', Courier, monospace !important;
}
header[data-testid="stHeader"] { display: none !important; }
[data-testid="collapsedControl"] { display: none !important; }
section[data-testid="stSidebar"] { display: none !important; }
.block-container { padding-top: 16px !important; }

.topbar {
    background: #e8f0fe;
    border-bottom: 2px solid #4a7fcb;
    padding: 10px 20px;
    margin-bottom: 20px;
    font-family: 'Courier New', monospace;
    font-size: 13px;
    color: #1a3a6b;
}
.topbar strong { font-size: 15px; }

div[data-testid="metric-container"] {
    border: 1px solid #b0c8f0;
    border-radius: 4px;
    padding: 10px 14px;
    background: #f4f8ff;
}
div[data-testid="metric-container"] label {
    font-family: 'Courier New', monospace !important;
    font-size: 11px !important;
    color: #4a7fcb !important;
}
div[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-family: 'Courier New', monospace !important;
    color: #1a3a6b !important;
}

.stButton button {
    background: #4a7fcb !important;
    color: white !important;
    border: none !important;
    border-radius: 3px !important;
    font-size: 13px !important;
}
.stButton button:hover { background: #3366b3 !important; }

.result-box {
    background: #f4f8ff;
    border: 1px solid #4a7fcb;
    padding: 16px 20px;
    margin-bottom: 14px;
    border-radius: 4px;
}

.hero-banner {
    background: #e8f0fe;
    border-bottom: 2px solid #4a7fcb;
    padding: 32px 24px 28px;
    margin-bottom: 24px;
    font-family: 'Courier New', monospace;
}
.hero-title {
    font-size: 28px;
    font-weight: 700;
    color: #1a3a6b;
    margin-bottom: 8px;
}
.hero-subtitle {
    font-size: 13px;
    color: #3a5a9b;
    margin-bottom: 18px;
    line-height: 1.6;
    max-width: 680px;
}
.hero-badge {
    display: inline-block;
    background: #4a7fcb;
    color: white;
    font-size: 11px;
    padding: 3px 10px;
    border-radius: 3px;
    margin-right: 8px;
    font-family: 'Courier New', monospace;
}
.stat-row {
    display: flex;
    gap: 1px;
    background: #b0c8f0;
    border: 1px solid #b0c8f0;
    border-radius: 4px;
    overflow: hidden;
    margin-bottom: 24px;
}
.stat-cell {
    flex: 1;
    background: #f4f8ff;
    padding: 16px 20px;
    font-family: 'Courier New', monospace;
}
.stat-num {
    font-size: 26px;
    font-weight: 700;
    color: #4a7fcb;
    line-height: 1;
    margin-bottom: 4px;
}
.stat-label {
    font-size: 11px;
    color: #4a7fcb;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.stat-sub {
    font-size: 11px;
    color: #8a9abf;
    margin-top: 2px;
}
.section-title {
    font-size: 14px;
    font-weight: 700;
    color: #1a3a6b;
    font-family: 'Courier New', monospace;
    margin-bottom: 12px;
    border-left: 3px solid #4a7fcb;
    padding-left: 10px;
}
.feature-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 12px;
    margin-bottom: 24px;
}
.feature-card {
    background: #f4f8ff;
    border: 1px solid #b0c8f0;
    border-radius: 4px;
    padding: 16px;
    font-family: 'Courier New', monospace;
}
.feature-icon { font-size: 22px; margin-bottom: 8px; }
.feature-name { font-size: 13px; font-weight: 700; color: #1a3a6b; margin-bottom: 6px; }
.feature-desc { font-size: 12px; color: #4a6090; line-height: 1.55; }

.steps-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin-bottom: 24px;
}
.step-card {
    background: #f4f8ff;
    border: 1px solid #b0c8f0;
    border-radius: 4px;
    padding: 16px;
    font-family: 'Courier New', monospace;
    text-align: center;
}
.step-num { font-size: 11px; color: #4a7fcb; margin-bottom: 6px; letter-spacing: 0.05em; }
.step-name { font-size: 13px; font-weight: 700; color: #1a3a6b; margin-bottom: 6px; }
.step-desc { font-size: 12px; color: #4a6090; line-height: 1.5; }

.classes-grid {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 8px;
    margin-bottom: 24px;
}
.class-tile {
    background: #f4f8ff;
    border: 1px solid #b0c8f0;
    border-radius: 4px;
    padding: 12px;
    font-family: 'Courier New', monospace;
}
.class-name { font-size: 11px; font-weight: 700; color: #1a3a6b; margin-bottom: 3px; }
.class-hint { font-size: 11px; color: #6a80a0; }

.health-cards { display: flex; flex-direction: column; gap: 8px; }
.health-card {
    display: flex;
    align-items: center;
    gap: 12px;
    background: #f4f8ff;
    border: 1px solid #b0c8f0;
    border-radius: 4px;
    padding: 12px 16px;
    font-family: 'Courier New', monospace;
}
.health-dot { width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }
.health-card-title { font-size: 13px; font-weight: 700; color: #1a3a6b; margin-bottom: 2px; }
.health-card-desc  { font-size: 12px; color: #4a6090; line-height: 1.45; }
.health-badge {
    font-size: 10px;
    padding: 3px 8px;
    border-radius: 3px;
    font-family: 'Courier New', monospace;
    flex-shrink: 0;
}
.badge-poor     { background: #fee2e2; color: #b91c1c; }
.badge-moderate { background: #fef9c3; color: #92400e; }
.badge-healthy  { background: #dcfce7; color: #166534; }
</style>
""", unsafe_allow_html=True)

# ── TOPBAR ──
st.markdown("""
<div class="topbar">
    <strong>SentinelCrop · Satellite Intelligence</strong>
    &nbsp;|&nbsp; EuroSAT Crop Type Classification &amp; Vegetation Health
</div>
""", unsafe_allow_html=True)

# ── HERO ──
st.markdown("""
<div class="hero-banner">
    <div class="hero-title">See Your Land from Space</div>
    <div class="hero-subtitle">
        Upload a satellite image and instantly find out what's growing on the ground —
        crops, forests, rivers, or cities. Then check if your vegetation is thriving
        or under stress, all without any technical knowledge.
    </div>
    <span class="hero-badge">Sentinel-2</span>
    <span class="hero-badge">EuroSAT</span>
    <span class="hero-badge">TensorFlow</span>
</div>
""", unsafe_allow_html=True)

# ── QUICK NAV ──
col_a, col_b = st.columns(2)
with col_a:
    st.page_link("pages/dashboard.py", label="→ Open Dashboard", icon="🛰️")
with col_b:
    st.page_link("pages/dashboard.py", label="→ View Training Results", icon="📊")

st.markdown("---")

# ── STATS ──
st.markdown("""
<div class="stat-row">
    <div class="stat-cell">
        <div class="stat-num">27K</div>
        <div class="stat-label">Training Images</div>
        <div class="stat-sub">Real satellite scenes</div>
    </div>
    <div class="stat-cell">
        <div class="stat-num">10</div>
        <div class="stat-label">Land Types</div>
        <div class="stat-sub">Crops, forests, water &amp; more</div>
    </div>
    <div class="stat-cell">
        <div class="stat-num">82%</div>
        <div class="stat-label">Accuracy</div>
        <div class="stat-sub">Verified on unseen images</div>
    </div>
    <div class="stat-cell">
        <div class="stat-num">97%</div>
        <div class="stat-label">Best Prediction</div>
        <div class="stat-sub">Forest detection confidence</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ── FEATURES ──
st.markdown('<div class="section-title">01 · What This Tool Does</div>', unsafe_allow_html=True)
st.markdown("""
<div class="feature-grid">
    <div class="feature-card">
        <div class="feature-icon">🛰️</div>
        <div class="feature-name">Identify Land Cover</div>
        <div class="feature-desc">Drop in a satellite image and get an instant label — farmland, forest, river, or city block. Trained on 27,000 real images from space.</div>
    </div>
    <div class="feature-card">
        <div class="feature-icon">🌿</div>
        <div class="feature-name">Check Vegetation Health</div>
        <div class="feature-desc">See whether plants in a region are thriving, struggling, or in poor condition. Color-coded maps make it easy to spot problem areas.</div>
    </div>
    <div class="feature-card">
        <div class="feature-icon">🎨</div>
        <div class="feature-name">Enhanced Imagery</div>
        <div class="feature-desc">False-color view makes crops and green areas pop out clearly, even in imagery that looks dull to the naked eye.</div>
    </div>
    <div class="feature-card">
        <div class="feature-icon">📍</div>
        <div class="feature-name">Pinpoint Stress Zones</div>
        <div class="feature-desc">The health map highlights exactly which parts of a field or region need attention — great for early detection of drought or disease.</div>
    </div>
    <div class="feature-card">
        <div class="feature-icon">📊</div>
        <div class="feature-name">Clear Confidence Scores</div>
        <div class="feature-desc">Every prediction comes with a plain-language confidence score — e.g. "Forest · 97% sure." No jargon, just a straight answer.</div>
    </div>
    <div class="feature-card">
        <div class="feature-icon">⚡</div>
        <div class="feature-name">Instant Results</div>
        <div class="feature-desc">Upload your image and get results in seconds. The model runs locally — no waiting for cloud processing. Works offline once set up.</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ── HOW IT WORKS ──
st.markdown('<div class="section-title">02 · How It Works</div>', unsafe_allow_html=True)
st.markdown("""
<div class="steps-row">
    <div class="step-card">
        <div class="step-num">Step 01</div>
        <div class="step-name">Upload an Image</div>
        <div class="step-desc">Drag in any Sentinel-2 satellite patch — farmland, city, river, anything.</div>
    </div>
    <div class="step-card">
        <div class="step-num">Step 02</div>
        <div class="step-name">AI Scans It</div>
        <div class="step-desc">The model studies colors, patterns and textures the same way it studied 27,000 examples.</div>
    </div>
    <div class="step-card">
        <div class="step-num">Step 03</div>
        <div class="step-name">Get a Label</div>
        <div class="step-desc">See the land type — Crop, Forest, River, City — plus a confidence score in plain English.</div>
    </div>
    <div class="step-card">
        <div class="step-num">Step 04</div>
        <div class="step-name">View Health Map</div>
        <div class="step-desc">A color map shows which zones are healthy, developing, or under stress — at a glance.</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ── CLASSES ──
st.markdown('<div class="section-title">03 · 10 Land Types We Recognize</div>', unsafe_allow_html=True)

CLASS_INFO = [
    ("AnnualCrop",            "Seasonal farmland"),
    ("Forest",                "Dense tree cover"),
    ("HerbaceousVegetation",  "Grasslands & shrubs"),
    ("Highway",               "Roads & transport"),
    ("Industrial",            "Factories & warehouses"),
    ("Pasture",               "Grazing land"),
    ("PermanentCrop",         "Orchards & vineyards"),
    ("Residential",           "Housing & suburbs"),
    ("River",                 "Waterways & streams"),
    ("SeaLake",               "Open water bodies"),
]

tiles_html = ""
for cls, hint in CLASS_INFO:
    tiles_html += f"""
    <div class="class-tile">
        <div class="class-name">{cls}</div>
        <div class="class-hint">{hint}</div>
    </div>"""

st.markdown(f'<div class="classes-grid">{tiles_html}</div>', unsafe_allow_html=True)

st.markdown("---")

# ── VEGETATION HEALTH ──
st.markdown('<div class="section-title">04 · Is Your Vegetation Healthy?</div>', unsafe_allow_html=True)

col_left, col_right = st.columns(2)

with col_left:
    st.markdown("""
Using infrared light invisible to the human eye, the system reads how
actively plants are photosynthesizing. The result is a simple color map —
no expertise needed to understand what it's telling you.
""")

with col_right:
    st.markdown("""
<div class="health-cards">
    <div class="health-card">
        <div class="health-dot" style="background:#ef4444;"></div>
        <div>
            <div class="health-card-title">Poor Condition</div>
            <div class="health-card-desc">Bare soil, standing water, or crops under severe stress. Likely needs urgent attention.</div>
        </div>
        <div class="health-badge badge-poor">Alert</div>
    </div>
    <div class="health-card">
        <div class="health-dot" style="background:#eab308;"></div>
        <div>
            <div class="health-card-title">Developing</div>
            <div class="health-card-desc">Sparse or young vegetation. Could be early-season growth or areas recovering from stress.</div>
        </div>
        <div class="health-badge badge-moderate">Watch</div>
    </div>
    <div class="health-card">
        <div class="health-dot" style="background:#22c55e;"></div>
        <div>
            <div class="health-card-title">Healthy &amp; Thriving</div>
            <div class="health-card-desc">Dense, actively growing vegetation. Crops and forests in this zone are performing well.</div>
        </div>
        <div class="health-badge badge-healthy">Good</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("---")
st.caption("SentinelCrop · Charith Manujaya · EuroSAT · Sentinel-2 · TensorFlow")