import pandas as pd
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1. Load dataset
data = pd.read_csv("data/sleep_data.csv")

# 2. Clean column names
data.columns = data.columns.str.strip()
#printing column names in the dataset
print("Columns in dataset:")
print(data.columns.tolist())

# 3. Rename long column names to simple names
data.rename(columns={
    "Sleep Duration (hours)": "Sleep Duration",
    "Quality of Sleep (scale: 1-10)": "Quality of Sleep",
    "Physical Activity Level (minutes/day)": "Physical Activity Level",
    "Stress Level (scale: 1-10)": "Stress Level",
    "Heart Rate (bpm)": "Heart Rate"
}, inplace=True)

# 4. Select needed columns
required_columns = [
    "Gender",
    "Age",
    "Occupation",
    "Sleep Duration",
    "Physical Activity Level",
    "Stress Level",
    "BMI Category",
    "Heart Rate",
    "Daily Steps",
    "Sleep Disorder"
]

data = data[required_columns].copy()

# 5. Create target column from Sleep Disorder
def convert_sleep_quality(disorder):
    disorder = str(disorder).strip()

    if disorder == "None":
        return "Good"
    elif disorder == "Insomnia":
        return "Poor"
    elif disorder == "Sleep Apnea":
        return "Average"
    else:
        return "Average"

data["Sleep Quality"] = data["Sleep Disorder"].apply(convert_sleep_quality)

# Remove helper column
data.drop("Sleep Disorder", axis=1, inplace=True)

# 6. Encode categorical input columns
label_encoders = {}
categorical_columns = ["Gender", "Occupation", "BMI Category"]

for col in categorical_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col].astype(str))
    label_encoders[col] = le

# 7. Encode target column
target_encoder = LabelEncoder()
data["Sleep Quality"] = target_encoder.fit_transform(data["Sleep Quality"])

# 8. Split features and target
X = data.drop("Sleep Quality", axis=1)
y = data["Sleep Quality"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 9. Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# 10. Check accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# 11. Save model and encoders
os.makedirs("model", exist_ok=True)

with open("model/sleep_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("model/label_encoders.pkl", "wb") as f:
    pickle.dump(label_encoders, f)

with open("model/target_encoder.pkl", "wb") as f:
    pickle.dump(target_encoder, f)

print("Model trained and saved successfully!")