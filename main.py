import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

# Load the dataset
df = pd.read_csv('C:/Users/abhis/OneDrive/Desktop/anomaly-detection/anomaly-detection-isolation-forest/creditcard.csv')

# Display the first few rows (optional)
print(df.head())

# Use all features except 'Class' and 'Time' for anomaly detection
features = df.drop(['Class', 'Time'], axis=1)

# Create and train the Isolation Forest model
model = IsolationForest(n_estimators=100, contamination=0.001, random_state=42)
model.fit(features)

# Predict anomalies (-1 = anomaly, 1 = normal)
df['anomaly'] = model.predict(features)

# Count and print the number of anomalies
print("Detected anomalies:", (df['anomaly'] == -1).sum())

# Plot histogram of anomaly labels
plt.figure(figsize=(8, 5))
plt.hist(df['anomaly'], bins=3, edgecolor='black')
plt.title('Anomaly Detection Results')
plt.xlabel('Anomaly Label')
plt.ylabel('Count')
plt.xticks([-1, 1])
plt.grid(True)
plt.tight_layout()
plt.show()

# Scatter plot of V1 vs V2 highlighting anomalies
plt.figure(figsize=(10, 6))

# Separate normal and anomalous points
normal = df[df['anomaly'] == 1]
anomalies = df[df['anomaly'] == -1]

# Plot normal points
plt.scatter(normal['V1'], normal['V2'], c='blue', label='Normal', s=2, alpha=0.5)
# Plot anomalous points
plt.scatter(anomalies['V1'], anomalies['V2'], c='red', label='Anomaly', s=10, alpha=0.8)

plt.title('Anomaly Detection using Isolation Forest (V1 vs V2)')
plt.xlabel('V1')
plt.ylabel('V2')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
