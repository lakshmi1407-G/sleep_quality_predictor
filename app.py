from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model and encoders
with open("model/sleep_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("model/label_encoders.pkl", "rb") as f:
    label_encoders = pickle.load(f)

with open("model/target_encoder.pkl", "rb") as f:
    target_encoder = pickle.load(f)


def get_sleep_tip(result):
    if result == "Poor":
        return (
            "Your sleep quality appears to be poor, so you should start improving your daily routine step by step. "
            "Try to sleep and wake up at the same time every day so your body gets a regular sleep cycle. "
            "Reduce mobile phone or laptop use before bedtime because screen light can disturb your sleep. "
            "Do some physical activity like walking or simple exercise during the day to help your body feel relaxed at night. "
            "Try to lower your stress by doing meditation, deep breathing, or listening to calm music before sleep. "
            "Also avoid caffeine or heavy meals late at night, because they can reduce sleep comfort."
        )
    elif result == "Average":
        return (
            "Your sleep quality is average, which means it is okay but still needs some improvement. "
            "Try to maintain a proper sleep schedule and avoid sleeping too late on most days. "
            "Increasing your physical activity a little more can help your body feel healthier and improve rest. "
            "Keep your stress level under control by taking short breaks and relaxing your mind in the evening. "
            "Try to reduce unnecessary screen time before bed and keep your room calm and comfortable for sleeping. "
            "Small healthy changes done every day can slowly improve your sleep quality."
        )
    else:
        return (
            "Your sleep quality looks good, which means your current routine is helping you well. "
            "Continue following a regular sleep schedule and try not to disturb your bedtime habits. "
            "Maintain daily physical activity and keep your stress level as low as possible. "
            "Drink caffeine in moderation and avoid taking it too close to bedtime. "
            "A peaceful room, less screen time, and a healthy routine will help you protect your good sleep quality. "
            "Keep following these habits so your body and mind stay active, fresh, and healthy."
        )


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        gender = request.form["gender"]
        age = int(request.form["age"])
        occupation = request.form["occupation"]
        sleep_duration = float(request.form["sleep_duration"])
        physical_activity = int(request.form["physical_activity"])
        stress_level = int(request.form["stress_level"])
        heart_rate = int(request.form["heart_rate"])
        daily_steps = int(request.form["daily_steps"])

        # BMI removed from UI, so default value is used internally
        default_bmi = "Normal"

        # Encode categorical values
        gender_encoded = label_encoders["Gender"].transform([gender])[0]
        occupation_encoded = label_encoders["Occupation"].transform([occupation])[0]
        bmi_encoded = label_encoders["BMI Category"].transform([default_bmi])[0]

        # Prepare model input
        features = np.array([[
            gender_encoded,
            age,
            occupation_encoded,
            sleep_duration,
            physical_activity,
            stress_level,
            bmi_encoded,
            heart_rate,
            daily_steps
        ]])

        prediction = model.predict(features)[0]
        result = target_encoder.inverse_transform([prediction])[0]
        tip = get_sleep_tip(result)

        return render_template(
            "index.html",
            prediction_text=f"Predicted Sleep Quality: {result}",
            tip_text=tip
        )

    except Exception as e:
        return render_template(
            "index.html",
            prediction_text=f"Error: {str(e)}",
            tip_text="Please check the entered values and try again."
        )


if __name__ == "__main__":
    app.run(debug=True)