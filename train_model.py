import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("enhanced_anxiety_dataset.csv")

# Encode categorical columns
le = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col])

# Select the 10 input features used in the Streamlit app
selected_features = [
    "Age", "Sleep Hours", "Physical Activity (hrs/week)", "Caffeine Intake (mg/day)",
    "Alcohol Consumption (drinks/week)", "Stress Level (1-10)", "Heart Rate (bpm)",
    "Breathing Rate (breaths/min)", "Sweating Level (1-5)", "Diet Quality (1-10)"
]

# Input and target
X = df[selected_features]
y = pd.cut(df["Anxiety Level (1-10)"], bins=[0, 3, 6, 10], labels=[0, 1, 2])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model and selected features
joblib.dump((model, selected_features), "anxiety_rf_model.pkl")
print("✅ Model saved as 'anxiety_rf_model.pkl'")
