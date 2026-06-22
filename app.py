import streamlit as st
import numpy as np
import pickle

# Page config
st.set_page_config(
    page_title="Prepaid to Postpaid Migration Predictor",
    page_icon="📱",
    layout="wide"
)

# Load model
@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

model = load_model()

# styling
st.markdown("""
<style>
    .main { background-color: #f8fafc; }
    .block-container { padding-top: 2rem; }

    .header-box {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d6a9f 100%);
        padding: 2rem 2.5rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        color: white;
    }
    .header-box h1 { font-size: 1.9rem; font-weight: 700; margin: 0; }
    .header-box p  { font-size: 0.95rem; opacity: 0.85; margin: 0.4rem 0 0; }

    .section-label {
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: #64748b;
        margin-bottom: 0.6rem;
        margin-top: 1.2rem;
    }

    .result-migrate {
        background: linear-gradient(135deg, #065f46, #047857);
        color: white;
        padding: 1.8rem 2rem;
        border-radius: 12px;
        text-align: center;
    }
    .result-stay {
        background: linear-gradient(135deg, #7c2d12, #b45309);
        color: white;
        padding: 1.8rem 2rem;
        border-radius: 12px;
        text-align: center;
    }
    .result-label { font-size: 1rem; opacity: 0.85; }
    .result-value { font-size: 2.4rem; font-weight: 800; margin: 0.3rem 0; }
    .result-sub   { font-size: 0.85rem; opacity: 0.75; }

    .metric-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        text-align: center;
    }
    .metric-card .val { font-size: 1.6rem; font-weight: 700; color: #1e3a5f; }
    .metric-card .lbl { font-size: 0.78rem; color: #64748b; margin-top: 0.2rem; }

    .stButton > button {
        background: linear-gradient(135deg, #1e3a5f, #2d6a9f);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 8px;
        font-size: 1rem;
        font-weight: 600;
        width: 100%;
        cursor: pointer;
    }
    .stButton > button:hover { opacity: 0.9; }

    .insight-box {
        background: #f1f5f9;
        border-left: 4px solid #2d6a9f;
        padding: 0.9rem 1.1rem;
        border-radius: 0 8px 8px 0;
        margin-top: 1rem;
        font-size: 0.88rem;
        color: #334155;
    }
</style>
""", unsafe_allow_html=True)

#header
st.markdown("""
<div class="header-box">
    <h1> Fadhlan demo interview Prepaid → Postpaid Migration Predictor</h1>
    <p>Enter customer profile details to predict likelihood of migrating to a postpaid plan.</p>
</div>
""", unsafe_allow_html=True)

#model metrics
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown('<div class="metric-card"><div class="val">80%</div><div class="lbl">Model Accuracy</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown('<div class="metric-card"><div class="val">83%</div><div class="lbl">Precision</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown('<div class="metric-card"><div class="val">83%</div><div class="lbl">Recall</div></div>', unsafe_allow_html=True)
with c4:
    st.markdown('<div class="metric-card"><div class="val">XGBoost</div><div class="lbl">Algorithm</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# input
left, right = st.columns([2, 1], gap="large")

with left:
    # Demographics
    st.markdown('<div class="section-label">👤 Demographics</div>', unsafe_allow_html=True)
    d1, d2, d3 = st.columns(3)
    with d1:
        age = st.number_input("Age", min_value=18, max_value=80, value=30)
    with d2:
        tenure_months = st.number_input("Tenure (months)", min_value=1, max_value=120, value=12)
    with d3:
        region = st.selectbox("Region", ["Central", "Northern", "Southern", "Eastern", "Sabah/Sarawak"])

    # Top-up History
    st.markdown('<div class="section-label">💳 Top-Up History</div>', unsafe_allow_html=True)
    t1, t2, t3 = st.columns(3)
    with t1:
        topup_frequency = st.number_input("Top-up Frequency (per month)", min_value=0, max_value=30, value=4)
    with t2:
        topup_avg_amount = st.number_input("Avg Top-up Amount (RM)", min_value=0.0, max_value=200.0, value=30.0)
    with t3:
        topup_regularity = st.slider("Top-up Regularity Score", 0.0, 1.0, 0.5, help="0 = irregular, 1 = very regular")

    # Voice Usage
    st.markdown('<div class="section-label"> Voice Usage</div>', unsafe_allow_html=True)
    v1, v2 = st.columns(2)
    with v1:
        outgoing_mins = st.number_input("Outgoing Call Minutes/month", min_value=0.0, max_value=1000.0, value=120.0)
    with v2:
        incoming_mins = st.number_input("Incoming Call Minutes/month", min_value=0.0, max_value=1000.0, value=100.0)

    # Data & SMS
    st.markdown('<div class="section-label"> Data & SMS Activity</div>', unsafe_allow_html=True)
    ds1, ds2 = st.columns(2)
    with ds1:
        data_usage_mb = st.number_input("Data Usage (MB/month)", min_value=0.0, max_value=20000.0, value=2000.0)
    with ds2:
        sms_frequency = st.number_input("SMS Sent (per month)", min_value=0, max_value=500, value=30)

    # USSD & Roaming
    st.markdown('<div class="section-label">USSD & Roaming</div>', unsafe_allow_html=True)
    u1, u2 = st.columns(2)
    with u1:
        ussd_queries = st.number_input("USSD Queries (per month)", min_value=0, max_value=100, value=5)
    with u2:
        has_roamed = st.selectbox("Has Ever Roamed?", ["No", "Yes"])

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("🔍 Predict Migration Likelihood")

# prediction output 
with right:
    st.markdown('<div class="section-label"> Prediction Result</div>', unsafe_allow_html=True)

    if predict_btn:
        # Encode inputs
        roam_val = 1 if has_roamed == "Yes" else 0
        regions = ['Central', 'Eastern', 'Northern', 'Sabah/Sarawak', 'Southern']
        region_encoded = [1 if region == r else 0 for r in regions]

        features = np.array([[
            age, tenure_months,
            topup_frequency, topup_avg_amount, topup_regularity,
            outgoing_mins, incoming_mins,
            data_usage_mb,
            sms_frequency,
            ussd_queries,
            roam_val,
            *region_encoded
        ]])

        prob = model.predict_proba(features)[0][1]
        pred = model.predict(features)[0]
        pct  = round(prob * 100, 1)

        if pred == 1:
            st.markdown(f"""
            <div class="result-migrate">
                <div class="result-label">Migration Likelihood</div>
                <div class="result-value">{pct}%</div>
                <div class="result-sub"> Likely to Migrate to Postpaid</div>
            </div>
            """, unsafe_allow_html=True)

            # Insight
            insights = []
            if data_usage_mb > 3000:
                insights.append("High data consumption suggests postpaid unlimited plans would be attractive.")
            if topup_avg_amount > 50:
                insights.append("High top-up spend indicates willingness to pay for postpaid.")
            if tenure_months > 24:
                insights.append("Long tenure shows loyalty — good candidate for upgrade offer.")
            if has_roamed == "Yes":
                insights.append("Roaming history suggests need for better international coverage.")

            if insights:
                st.markdown(f'<div class="insight-box"> <b>Key Signals:</b><br>{"<br>".join(f"• {i}" for i in insights)}</div>', unsafe_allow_html=True)

        else:
            st.markdown(f"""
            <div class="result-stay">
                <div class="result-label">Migration Likelihood</div>
                <div class="result-value">{pct}%</div>
                <div class="result-sub"> Unlikely to Migrate to Postpaid</div>
            </div>
            """, unsafe_allow_html=True)

            insights = []
            if data_usage_mb < 500:
                insights.append("Low data usage — current prepaid plan likely sufficient.")
            if topup_avg_amount < 15:
                insights.append("Low top-up amount suggests price sensitivity.")
            if tenure_months < 6:
                insights.append("Short tenure — customer may still be evaluating the service.")

            if insights:
                st.markdown(f'<div class="insight-box"> <b>Key Signals:</b><br>{"<br>".join(f"• {i}" for i in insights)}</div>', unsafe_allow_html=True)

        # Probability bar
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**Confidence Score**")
        st.progress(prob)
        st.caption(f"Model is {pct}% confident this customer will migrate.")

    else:
        st.info("Fill in the customer details on the left and click **Predict** to see the result.")

        st.markdown("""
        <div class="insight-box">
        <b>How to use:</b><br>
        • Fill in the customer's demographic and behavioral data<br>
        • Click Predict to get migration likelihood<br>
        • Use the result to prioritize outreach campaigns
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.caption("Demo version · Built with XGBoost · Trained on synthetic data reflecting real feature engineering")
