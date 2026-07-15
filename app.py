import streamlit as st
import pandas as pd

from model import load_data, build_default_input, train_model, predict, CATEGORICAL_COLUMNS, NUMERICAL_COLUMNS


@st.cache_data(show_spinner=False)
def get_data() -> pd.DataFrame:
    return load_data("credit.csv")


@st.cache_data(show_spinner=False)
def get_model():
    df = get_data()
    pipeline, results = train_model(df)
    return pipeline, results


def render_sidebar(df: pd.DataFrame) -> dict:
    st.sidebar.header("Applicant profile")
    model_inputs = {}
    for feature in CATEGORICAL_COLUMNS:
        options = sorted(df[feature].dropna().unique())
        model_inputs[feature] = st.sidebar.selectbox(feature, options, index=0)
    for feature in NUMERICAL_COLUMNS:
        value = float(df[feature].median(skipna=True))
        min_val = float(df[feature].min(skipna=True))
        max_val = float(df[feature].max(skipna=True))
        step = max(1.0, (max_val - min_val) / 100)
        if feature in ["Age", "Num Credit Cards", "Num Late Payments", "Num Inquiries 6M"]:
            model_inputs[feature] = st.sidebar.number_input(feature, min_value=int(min_val), max_value=int(max_val), value=int(value), step=1)
        else:
            model_inputs[feature] = st.sidebar.number_input(feature, min_value=min_val, max_value=max_val, value=value, step=step, format="%.2f")
    return model_inputs


def main():
    st.set_page_config(page_title="Credit Card Approval Predictor", layout="wide")
    st.title("Credit Card Approval Prediction")
    st.write(
        "Use the applicant profile inputs in the sidebar to predict whether a credit card application is likely to be approved. "
        "The model is trained on the provided dataset and evaluates performance on a holdout split."
    )

    df = get_data()
    pipeline, results = get_model()

    with st.expander("Dataset summary"):
        st.write(df.describe(include="all"))
        st.write(df[CATEGORICAL_COLUMNS + NUMERICAL_COLUMNS + ["Approved"]].head(10))

    st.subheader("Model performance")
    st.metric("Accuracy", f"{results['accuracy']:.3f}")
    st.metric("ROC AUC", f"{results['roc_auc']:.3f}")
    st.text_area("Classification report", results["report"], height=220)

    st.sidebar.title("Predict an application")
    model_inputs = render_sidebar(df)

    if st.sidebar.button("Predict approval"):
        prediction = predict(pipeline, model_inputs)
        status = "Approved" if prediction["label"] == 1 else "Declined"
        st.markdown(f"### Prediction: **{status}**")
        st.markdown(f"**Approval probability:** {prediction['probability'] * 100:.1f}%")
        st.write("#### Input profile")
        st.dataframe(prediction["input"], use_container_width=True)

    st.sidebar.markdown("---")
    st.sidebar.write("Data count: ", df.shape[0])
    st.sidebar.write("Positive approvals: ", int(df["Approved"].sum()))


if __name__ == "__main__":
    main()
