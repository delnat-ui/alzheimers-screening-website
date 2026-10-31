import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(page_title="Cognitive Screening Research Prototype", page_icon="🧠", layout="wide")
MODEL_PATH = Path(__file__).parent / "alzheimers_screening_model.joblib"
st.title("🧠 Cognitive Screening Research Prototype")
st.caption("Explainable Machine Learning-Based Preliminary Screening — Academic Demonstration")
st.warning("Research prototype only. This is NOT a medical device and does NOT diagnose Alzheimer's disease or any other condition. Do not enter real patient information or use results for healthcare decisions.")

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

st.info(f"Loaded model: **{model_name}** · Required input fields: **{len(feature_columns)}** · Target: **{target_column}**")
st.markdown("Fill in the demonstration form below. Choose readable options wherever available. The app converts choices to the numeric codes expected by the trained model.")
st.caption("The underlying dataset is synthetic. This form is for demonstrating the model workflow, not for assessing real people.")

# Category code mappings used by this dataset. Numeric codes are passed to the model internally.
choice_maps = {
    "Gender": {"Male": 0, "Female": 1},
    "Ethnicity": {"Caucasian": 0, "African American": 1, "Asian": 2, "Other": 3},
    "EducationLevel": {"None": 0, "High School": 1, "Bachelor's": 2, "Higher": 3},
}
binary_fields = {
    "Smoking", "FamilyHistoryAlzheimers", "CardiovascularDisease", "Diabetes",
    "Depression", "HeadInjury", "Hypertension", "MemoryComplaints",
    "BehavioralProblems", "Confusion", "Disorientation", "PersonalityChanges",
    "DifficultyCompletingTasks", "Forgetfulness"
}
labels = {
    "FamilyHistoryAlzheimers": "Family history of Alzheimer's",
    "CardiovascularDisease": "Cardiovascular disease",
    "MemoryComplaints": "Memory complaints",
    "BehavioralProblems": "Behavioral problems",
    "DifficultyCompletingTasks": "Difficulty completing tasks",
}
groups = {
    "Demographic information": ["Age", "Gender", "Ethnicity", "EducationLevel"],
    "Lifestyle information": ["BMI", "Smoking", "AlcoholConsumption", "PhysicalActivity", "DietQuality", "SleepQuality"],
    "Medical history": ["FamilyHistoryAlzheimers", "CardiovascularDisease", "Diabetes", "Depression", "HeadInjury", "Hypertension"],
    "Blood pressure and cholesterol": ["SystolicBP", "DiastolicBP", "CholesterolTotal", "CholesterolLDL", "CholesterolHDL", "CholesterolTriglycerides"],
    "Cognitive and functional measures": ["MMSE", "FunctionalAssessment", "MemoryComplaints", "BehavioralProblems", "ADL", "Confusion", "Disorientation", "PersonalityChanges", "DifficultyCompletingTasks", "Forgetfulness"],
}
known = set(feature_columns)
groups = {k: [f for f in v if f in known] for k, v in groups.items()}
assigned = {f for v in groups.values() for f in v}
if assigned != known:
    groups["Other model inputs"] = [f for f in feature_columns if f not in assigned]

st.subheader("Demonstration input form")
st.caption("All fields are required. Measurements remain numeric; categorical fields use dropdowns or Yes/No selections.")
with st.form("screening_input_form"):
    values = {}
    for group_title, names in groups.items():
        if not names:
            continue
        st.markdown(f"### {group_title}")
        cols = st.columns(2)
        for index, feature in enumerate(names):
            with cols[index % 2]:
                label = labels.get(feature, feature.replace("_", " "))
                if feature in choice_maps:
                    options = list(choice_maps[feature].keys())
                    selected = st.selectbox(label, options, key=f"field_{feature}")
                    values[feature] = choice_maps[feature][selected]
                elif feature in binary_fields:
                    selected = st.selectbox(label, ["No", "Yes"], key=f"field_{feature}")
                    values[feature] = 0 if selected == "No" else 1
                else:
                    help_text = "Enter the numeric measurement/score using the dataset's units."
                    values[feature] = st.number_input(label, value=0.0, format="%.4f", help=help_text, key=f"field_{feature}")
        st.divider()
    submitted = st.form_submit_button("Run preliminary model estimate", type="primary")

if submitted:
    input_df = pd.DataFrame([[values[name] for name in feature_columns]], columns=feature_columns)
    try:
        prediction = pipeline.predict(input_df)[0]
        label_encoder = bundle.get("label_encoder")
        display_prediction = str(prediction)
        if label_encoder is not None:
            try:
                display_prediction = str(label_encoder.inverse_transform([prediction])[0])
            except Exception:
                pass
        st.subheader("Model output")
        st.metric("Predicted dataset class", display_prediction)
        if hasattr(pipeline, "predict_proba"):
            probabilities = pipeline.predict_proba(input_df)[0]
            classes = list(pipeline.classes_)
            decoded_classes = []
            for c in classes:
                try:
                    decoded_classes.append(str(label_encoder.inverse_transform([c])[0]) if label_encoder is not None else str(c))
                except Exception:
                    decoded_classes.append(str(c))
            probability_table = pd.DataFrame({"Dataset class": decoded_classes, "Model-estimated probability": [f"{v*100:.2f}%" for v in probabilities]})
            st.dataframe(probability_table, use_container_width=True, hide_index=True)
        st.warning("This is only a model output based on synthetic research data. It is not a person's medical risk, screening result, or diagnosis. Do not use it for clinical decisions.")
        st.download_button("Download demonstration result (CSV)", data=input_df.assign(Model_Predicted_Class=str(display_prediction)).to_csv(index=False).encode("utf-8"), file_name="research_demo_model_output.csv", mime="text/csv")
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        st.caption("Please check that the Streamlit environment uses scikit-learn 1.6.1 as specified in requirements.txt.")

with st.expander("About this research prototype"):
    st.write("The uploaded Random Forest model was trained on a synthetic dataset. Synthetic data can demonstrate a technical workflow, but cannot establish clinical validity, real-world accuracy, or suitability for screening patients.")
    st.write("Any future patient-facing research would require an ethically approved clinical dataset, validated assessment instruments, external validation, subgroup evaluation, privacy safeguards, and qualified clinical oversight.")
st.caption("Academic demonstration only · Do not enter real patient data.")
