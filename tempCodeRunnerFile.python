

import streamlit as st
import numpy as np
import pandas as pd
import pickle
from sklearn.datasets import load_iris
import seaborn as sns
import matplotlib.pyplot as plt

# Load model
with open("logistic_regression_model.pkl", "rb") as file:
    model = pickle.load(file)

# Load iris dataset for visuals
iris = load_iris()
class_names = iris.target_names
feature_names = iris.feature_names
df_iris = pd.DataFrame(iris.data, columns=feature_names)
df_iris["species"] = [class_names[i] for i in iris.target]

# Streamlit layout
st.set_page_config(page_title="🌸 Iris Species Predictor", layout="wide")
st.title("🌸 Iris Flower Species Predictor")
st.markdown("This app predicts the species of Iris flowers using a trained model.")

# Tabs
tab1, tab2 = st.tabs(["🔍 Single Prediction", "📂 Batch Prediction (CSV)"])

# ---------------------------------------
# 🔍 SINGLE PREDICTION
# ---------------------------------------
with tab1:
    st.header("🔍 Predict from Input Values")
    sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.8)
    sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.0)
    petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 4.3)
    petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 1.3)

    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    if st.button("Predict Species"):
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0]

        st.success(f"Predicted species: *{class_names[prediction].capitalize()}*")
        st.subheader("📊 Prediction Probabilities")
        for i, prob in enumerate(probability):
            st.write(f"{class_names[i].capitalize()}: *{round(prob * 100, 2)}%*")

# ---------------------------------------
# 📂 BATCH PREDICTION
# ---------------------------------------
with tab2:
    st.header("📂 Upload CSV for Batch Prediction")
    uploaded_file = st.file_uploader("Upload a CSV file with 4 columns", type=["csv"])

    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)

        if batch_df.shape[1] != 4:
            st.error("❌ The CSV file must have exactly 4 columns (sepal and petal measurements).")
        else:
            st.write("✅ File successfully uploaded!")
            st.dataframe(batch_df.head())

            try:
                preds = model.predict(batch_df)
                probs = model.predict_proba(batch_df)

                results_df = batch_df.copy()
                results_df["Predicted_Species"] = [class_names[i] for i in preds]
                results_df["Probability (%)"] = np.max(probs, axis=1) * 100

                st.subheader("📄 Prediction Results")
                st.dataframe(results_df)

                # Download button
                csv_data = results_df.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download Results as CSV", csv_data, "iris_predictions.csv", "text/csv")

            except Exception as e:
                st.error(f"Prediction failed: {e}")

# ---------------------------------------
# 📊 VISUALIZATIONS
# ---------------------------------------
st.header("📊 Data Visualizations")

st.subheader("🌈 Pairplot of Iris Dataset")
fig1 = sns.pairplot(df_iris, hue="species", diag_kind="hist")
st.pyplot(fig1)

st.subheader("📌 Species Distribution")
fig2, ax2 = plt.subplots()
sns.countplot(data=df_iris, x="species", ax=ax2, palette="Set2")
ax2.set_title("Species Count")
st.pyplot(fig2)

if hasattr(model, "feature_importances_"):
    st.subheader("🌟 Feature Importance")
    fig3, ax3 = plt.subplots()
    sns.barplot(x=model.feature_importances_, y=feature_names, ax=ax3, palette="Blues_r")
    ax3.set_title("Feature Importance")
    st.pyplot(fig3)

