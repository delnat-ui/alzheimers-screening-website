import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(
    page_title="Cognitive Screening Research Prototype",
    page_icon="🧠",
    layout="wide"
)

MODEL_PATH = Path(__file__).parent / "alzheimers_screening_model.joblib"

st.title("🧠 Cognitive Screening Research Prototype")
st.caption("Explainable Machine Learning-Based Preliminary Screening — Academic Demonstration")

st.warning(
    "Research prototype only. This application is NOT a medical device and does NOT "
    "diagnose Alzheimer's disease or any other condition. Do not enter real patient "
    "information. Any result must not be used to make healthcare decisions."
)

@st.cache_resource
def load_bundle():
    return joblib.load(MODEL_PATH)

if not MODEL_PATH.exists():
    st.error("Model file not found. Keep alzheimers_screening_model.joblib in the same folder as app.py.")
    st.stop()

try:
    bundle = load_bundle()
    pipeline = bundle["pipeline"]
    feature_columns = bundle["feature_columns"]
    model_name = bundle.get("model_name", "Trained model")
    target_column = bundle.get("target_column", "Diagnosis")
except Exception as exc:
    st.error(f"Could not load the model: {exc}")
    st.stop()

st.info(
    f"Loaded model: **{model_name}** · Required input fields: **{len(feature_columns)}** · "
    f"Target column: **{target_column}**"
)

st.markdown(
    "This form collects the same dataset variables used when training the uploaded model. "
    "It is not a validated clinical questionnaire. Enter values only when you know their "
    "meaning and coding from the dataset codebook."
)

groups = {
    "Demographic information": [
        "Age", "Gender", "Ethnicity", "EducationLevel"
    ],
    "Lifestyle information": [
        "BMI", "Smoking", "AlcoholConsumption", "PhysicalActivity",
        "DietQuality", "SleepQuality"
    ],
    "Medical history": [
        "FamilyHistoryAlzheimers", "CardiovascularDisease", "Diabetes",
        "Depression", "HeadInjury", "Hypertension"
    ],
    "Blood pressure and cholesterol": [
        "SystolicBP", "DiastolicBP", "CholesterolTotal",
        "CholesterolLDL", "CholesterolHDL", "CholesterolTriglycerides"
    ],
    "Cognitive and functional measures": [
        "MMSE", "FunctionalAssessment", "MemoryComplaints",
        "BehavioralProblems", "ADL", "Confusion", "Disorientation",
        "PersonalityChanges", "DifficultyCompletingTasks", "Forgetfulness"
    ],
}

# Keep the app aligned exactly with the saved model's feature order.
known = set(feature_columns)
groups = {title: [name for name in names if name in known] for title, names in groups.items()}
assigned = {name for names in groups.values() for name in names}
if assigned != known:
    groups["Other model inputs"] = [name for name in feature_columns if name not in assigned]

st.subheader("Enter model input values")
st.caption("All fields are required. Use the original dataset's coding and units. Do not guess unknown values.")

with st.form("screening_input_form"):
    values = {}
    for group_title, names in groups.items():
        if not names:
            continue
        st.markdown(f"**{group_title}**")
        cols = st.columns(2)
        for index, feature in enumerate(names):
            with cols[index % 2]:
                values[feature] = st.text_input(
                    feature,
                    value="",
                    placeholder="Enter numeric value",
                    help=f"Enter the numeric value for the dataset field '{feature}'. Check the dataset codebook for units and category coding."
                )
        st.divider()

    submitted = st.form_submit_button("Run preliminary model estimate", type="primary")

if submitted:
    missing = [name for name in feature_columns if not values.get(name, "").strip()]
    invalid = []
    parsed = {}
    for name in feature_columns:
        raw = values.get(name, "").strip()
        if not raw:
            continue
        try:
            parsed[name] = float(raw)
        except ValueError:
            invalid.append(name)

    if missing:
        st.error("Please complete all fields. Missing: " + ", ".join(missing))
    elif invalid:
        st.error("Please enter numeric values for: " + ", ".join(invalid))
    else:
        input_df = pd.DataFrame([[parsed[name] for name in feature_columns]], columns=feature_columns)
        try:
            prediction = pipeline.predict(input_df)[0]
            st.subheader("Model output")
            st.metric("Predicted target class", str(prediction))
            if hasattr(pipeline, "predict_proba"):
                probabilities = pipeline.predict_proba(input_df)[0]
                classes = list(pipeline.classes_)
                probability_table = pd.DataFrame({
                    "Model class": [str(c) for c in classes],
                    "Model-estimated probability": [f"{p * 100:.2f}%" for p in probabilities]
                })
                st.dataframe(probability_table, use_container_width=True, hide_index=True)
            st.warning(
                "This is a model output on synthetic research data, not an individual's medical "
                "risk or diagnosis. The class coding must be interpreted using the dataset "
                "documentation. Do not use this result for clinical decisions."
            )
            st.download_button(
                "Download demonstration result (CSV)",
                data=input_df.assign(Model_Predicted_Class=str(prediction)).to_csv(index=False).encode("utf-8"),
                file_name="research_demo_model_output.csv",
                mime="text/csv"
            )
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
            st.caption("Check that the installed scikit-learn version matches the version used to save the model.")

with st.expander("About this research prototype"):
    st.write(
        "The uploaded model bundle contains a preprocessing pipeline and a Random Forest "
        "classifier trained on a synthetic dataset. Synthetic data is useful for demonstrating "
        "a technical workflow, but it cannot establish clinical validity, real-world accuracy, "
        "or suitability for screening patients."
    )
    st.write(
        "Before any real-world or patient-facing use, the research would require an appropriate "
        "ethically approved clinical dataset, validated assessment instruments, external "
        "validation, subgroup evaluation, privacy safeguards, and qualified clinical oversight."
    )

st.caption("For academic demonstration only · No patient data should be entered or stored.")
