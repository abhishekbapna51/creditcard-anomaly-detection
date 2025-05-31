import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
import io

st.title("Credit Card Anomaly Detection with Isolation Forest")

# File upload
uploaded_file = st.file_uploader("Upload your credit card CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview")
    st.dataframe(df.head())

    # Select features (exclude 'Class' and 'Time' if present)
    features = df.drop(columns=[col for col in ['Class', 'Time'] if col in df.columns])

    # Sidebar parameters
    st.sidebar.header("Model Parameters")
    contamination = st.sidebar.number_input("Contamination (anomaly proportion)", min_value=0.0001, max_value=0.05, value=0.001, step=0.0001, format="%.4f")

    n_estimators = st.sidebar.slider("Number of Trees (n_estimators)", 50, 300, 100, step=10)
    random_state = 42

    # Train model button
    if st.sidebar.button("Run Anomaly Detection"):
        model = IsolationForest(n_estimators=n_estimators, contamination=contamination, random_state=random_state)
        model.fit(features)

        # Predict anomalies
        df['anomaly'] = model.predict(features)
        df['anomaly_label'] = df['anomaly'].apply(lambda x: 'Anomaly' if x == -1 else 'Normal')

        st.write("### Anomaly Detection Results")
        st.write(f"Total Records: {len(df)}")
        st.write(f"Detected Anomalies: {(df['anomaly'] == -1).sum()}")

        # Show anomalies table
        st.write("#### Sample Anomalies")
        st.dataframe(df[df['anomaly'] == -1].head(10))

        # Scatter plot example: pick any two features to visualize
        numeric_cols = features.select_dtypes(include='number').columns.tolist()
        x_axis = st.selectbox("X-axis feature for scatter plot", numeric_cols, index=0)
        y_axis = st.selectbox("Y-axis feature for scatter plot", numeric_cols, index=1)

        fig, ax = plt.subplots(figsize=(10, 6))
        colors = df['anomaly'].map({1: 'blue', -1: 'red'})
        ax.scatter(df[x_axis], df[y_axis], c=colors, alpha=0.5)
        ax.set_xlabel(x_axis)
        ax.set_ylabel(y_axis)
        ax.set_title("Anomaly Detection Scatter Plot\nBlue: Normal, Red: Anomaly")
        st.pyplot(fig)

        # Export anomalies CSV
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_bytes = csv_buffer.getvalue().encode()

        st.download_button(
            label="Download full results as CSV",
            data=csv_bytes,
            file_name="anomaly_detection_results.csv",
            mime="text/csv"
        )
else:
    st.info("Please upload a credit card dataset CSV file to start.")


footer = """
<style>
.footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    text-align: center;
    font-size: 12px;
    color: gray;
}
</style>
<div class="footer">
© 2025 Abhishek. All rights reserved.
</div>
"""
st.markdown(footer, unsafe_allow_html=True)
