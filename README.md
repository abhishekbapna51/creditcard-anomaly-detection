#Credit Card Anomaly Detection using Isolation Forest
This project implements an unsupervised anomaly detection system for identifying fraudulent credit card transactions. It uses the Isolation Forest algorithm and provides an interactive Streamlit application for data upload, visualization, and exporting results.

Overview
Credit card fraud is rare and difficult to detect using traditional supervised learning. This project uses Isolation Forest, an unsupervised model that identifies unusual patterns by isolating data points that behave differently from the majority. The Streamlit interface allows users to run anomaly detection on their own datasets and analyze results visually.

Features
Upload any credit card transaction CSV file.
Automatic feature selection (excludes columns like Class and Time if present).
Adjustable model parameters (contamination rate, number of trees).
Generates anomaly predictions and summary statistics.
Displays sample anomalous transactions.
Interactive scatter plot using selected features.
Option to download full results as a CSV file.
Simple and responsive Streamlit UI.

How It Works
Isolation Forest isolates data points by recursively splitting features. Anomalies require fewer splits and are classified as outliers. The model outputs:
1 for normal transactions
-1 for anomalous transactions

Project Structure
app.py                     # Streamlit application
requirements.txt           # Dependencies
creditcard.csv (optional)  # Example dataset

Setup and Usage
Install dependencies:
pip install -r requirements.txt
Run the Streamlit app:
streamlit run app.py
Upload a CSV file containing numerical transaction features.
Adjust model parameters as needed and run anomaly detection.

Core Model Logic
from sklearn.ensemble import IsolationForest
features = df.drop(['Class', 'Time'], axis=1, errors='ignore')
model = IsolationForest(
    n_estimators=100,
    contamination=0.001,
    random_state=42
)
model.fit(features)
df['anomaly'] = model.predict(features)

Live Application:
https://creditcard-anomaly-detection-vebfssgshcanmwt9sefdw7.streamlit.app/

Author
Abhishek Bapna
GitHub: https://github.com/abhishekbapna51
Portfolio: my-portfolio-abhishek-bapnas-projects.vercel.app
