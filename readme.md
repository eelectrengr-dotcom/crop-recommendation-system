🌱 Crop Recommendation System

📌 Overview

This project recommends the most suitable crop based on soil nutrients and environmental conditions using Machine Learning.

🚀 Features

Predict best crop using:

* Random Forest
* XGBoost
* FastAPI backend
* Streamlit interactive UI

📊 Input Features

* Nitrogen (N)
* Phosphorus (P)
* Potassium (K)
* Temperature
* Humidity
* pH
* Rainfall

🧠 Model Training

Run:

bash
python train.py


⚡ Run Backend API

bash
uvicorn api:app --reload


🎨 Run Frontend

bash
streamlit run app.py


📂 Project Structure

Crop-Recommendation-System/
│
├── data/
├── models/
├── train.py
├── api.py
├── app.py
├── requirements.txt
└── README.md

📌 Output

The system predicts the most suitable crop for given soil conditions.

💡 Skills Gained

* Agricultural AI
* Machine Learning Models
* API Development (FastAPI)
* Web App Deployment (Streamlit)

📎 Author

FATIMA NOOR (Machine Learning Internship - Week 5 Project)
