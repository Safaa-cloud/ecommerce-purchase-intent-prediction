import joblib
import pandas as pd
import streamlit as st
from pathlib import Path

MODEL = Path(__file__).parent / "models" / "purchase_intent_pipeline.joblib"
st.set_page_config(page_title="Purchase Intent Predictor", page_icon="🛍️")
st.title("🛍️ Will This Visitor Buy?")
st.write("Estimate purchase probability from browsing-session behavior.")
st.info("Educational prototype using historical session aggregates. Not a validated live prediction or automatic discount policy.")
if not MODEL.exists():
    st.error("Model not found. Run the notebook first to create models/purchase_intent_pipeline.joblib.")
    st.stop()
pipe = joblib.load(MODEL)
EXPECTED_FEATURES = [
    "Administrative", "Administrative_Duration", "Informational",
    "Informational_Duration", "ProductRelated", "ProductRelated_Duration",
    "BounceRates", "ExitRates", "SpecialDay", "Month", "OperatingSystems",
    "Browser", "Region", "TrafficType", "VisitorType", "Weekend",
]
if list(getattr(pipe, "feature_names_in_", [])) != EXPECTED_FEATURES:
    st.error("The saved model feature schema does not match the app inputs. Retrain the model with src/train.py.")
    st.stop()
with st.form("session"):
    a,b = st.columns(2)
    with a:
        admin = st.number_input("Administrative pages", 0, 100, 1)
        admin_d = st.number_input("Administrative duration (sec)", 0.0, 100000.0, 30.0)
        info = st.number_input("Informational pages", 0, 100, 0)
        info_d = st.number_input("Informational duration (sec)", 0.0, 100000.0, 0.0)
        product = st.number_input("Product-related pages", 0, 1000, 10)
        product_d = st.number_input("Product-related duration (sec)", 0.0, 200000.0, 300.0)
        bounce = st.number_input("Bounce rate", 0.0, 1.0, 0.02, 0.01)
        exit_rate = st.number_input("Exit rate", 0.0, 1.0, 0.04, 0.01)
    with b:
        special = st.select_slider("Special day closeness", [0.0,0.2,0.4,0.6,0.8,1.0], value=0.0)
        month = st.selectbox("Month", ["Feb","Mar","May","June","Jul","Aug","Sep","Oct","Nov","Dec"])
        os_code = st.selectbox("Operating system code", list(range(1,9)), index=1)
        browser = st.selectbox("Browser code", list(range(1,14)))
        region = st.selectbox("Region code", list(range(1,10)))
        traffic = st.selectbox("Traffic type code", list(range(1,21)), index=2)
        visitor = st.selectbox("Visitor type", ["Returning_Visitor","New_Visitor","Other"])
        weekend = st.checkbox("Weekend session")
    submit = st.form_submit_button("Predict", type="primary")
if submit:
    row = pd.DataFrame([{
        "Administrative":admin,"Administrative_Duration":admin_d,
        "Informational":info,"Informational_Duration":info_d,
        "ProductRelated":product,"ProductRelated_Duration":product_d,
        "BounceRates":bounce,"ExitRates":exit_rate,"SpecialDay":special,
        "Month":month,"OperatingSystems":os_code,"Browser":browser,
        "Region":region,"TrafficType":traffic,"VisitorType":visitor,"Weekend":weekend
    }])
    try:
        p = float(pipe.predict_proba(row)[0, 1])
    except (ValueError, KeyError, TypeError) as exc:
        st.error(f"Could not score this session. Check the input values and model artifact. Details: {exc}")
        st.stop()
    st.metric("Estimated purchase probability", f"{p:.1%}")
    if p >= 0.60:
        recommendation = "No incentive by default; the estimated purchase probability is already relatively high."
    elif p >= 0.30:
        recommendation = "Consider optional product guidance or a chat assistant; avoid an automatic discount."
    else:
        recommendation = "Consider clearer product information or a low-cost help prompt."
    st.subheader("Suggested engagement")
    st.write(recommendation)
    st.caption("Thresholds are illustrative, not optimized business cut-offs. Validate calibration, costs and incremental lift with an A/B test.")
