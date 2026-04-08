# 🌙 Sleep Quality Predictor

## 📌 Project Overview

The **Sleep Quality Predictor** is a Machine Learning-based web application that analyzes lifestyle and physiological factors to predict an individual's sleep quality. The system uses a **Random Forest Classifier** trained on the *Sleep Health and Lifestyle Dataset* to evaluate various inputs such as age, sleep duration, stress level, physical activity, heart rate, and daily steps.

The application provides a simple web interface where users can enter their personal health data and receive a prediction of their sleep quality along with helpful recommendations to improve their sleep habits.

This project demonstrates the integration of **Machine Learning, Data Analysis, and Web Development** to build a practical solution for personal health monitoring.

---

## 🚀 Features

* Predicts **sleep quality** using Machine Learning
* Interactive **web interface built with Flask**
* Takes multiple **health and lifestyle inputs**
* Provides **personalized sleep improvement tips**
* User-friendly **Input and Output UI**
* Dark **night sky themed interface** with animated stars and moon

---

## 🧠 Machine Learning Model

The prediction model is built using:

* **Algorithm:** Random Forest Classifier
* **Dataset:** Sleep Health and Lifestyle Dataset
* **Accuracy:** ~92%

### Input Features

The model evaluates the following features:

* Gender
* Age
* Occupation
* Sleep Duration
* Physical Activity Level
* Stress Level
* Heart Rate
* Daily Steps
* BMI Category (used internally by the model)

---

## 🛠️ Technologies Used

### Programming Language

* Python

### Libraries & Frameworks

* Flask
* Pandas
* NumPy
* Scikit-learn
* Pickle

### Frontend

* HTML
* CSS
* Custom Night Sky Animation UI

---

## 📂 Project Structure

```
sleep_quality_predictor
│
├── app.py
│
├── model
│   ├── sleep_model.pkl
│   ├── label_encoders.pkl
│   └── target_encoder.pkl
│
├── data
│   └── sleep_health_dataset.csv
│
├── templates
│   └── index.html
│
├── static
│
├── train_model.py
│
└── README.md
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository

```
git clone https://github.com/yourusername/sleep-quality-predictor.git
```

### 2️⃣ Navigate to Project Folder

```
cd sleep-quality-predictor
```

### 3️⃣ Install Required Libraries

```
pip install flask pandas numpy scikit-learn
```

### 4️⃣ Run the Application

```
python app.py
```

### 5️⃣ Open in Browser

```
http://127.0.0.1:5000/
```

---

## 📊 How It Works

1. The user enters personal health information in the web interface.
2. The input data is processed and encoded using trained label encoders.
3. The trained **Random Forest model** predicts the sleep quality category.
4. The application displays:

   * Predicted sleep quality
   * Personalized improvement tips

---


---

## 🎯 Future Improvements

* Integration with **wearable health devices**
* Development of a **mobile application**
* Implementation of **Deep Learning models**
* Real-time **sleep tracking and analytics**
* Cloud deployment for public access

---


