# Medical Device Failure Prediction

[Live Demo on Render](https://medical-device-failure-precition.onrender.com/)

---

## Table of Contents

- [Overview](#overview)  
- [Features](#features)  
- [Tech Stack](#tech-stack)  
- [Installation](#installation)  
- [Usage](#usage)  
- [Project Structure](#project-structure)  
- [How It Works](#how-it-works)  
- [Contributing](#contributing)  
- [License](#license)  

---

## Overview

**Medical Device Failure Prediction** is a web-based application designed to help healthcare professionals and manufacturers **predict the risk of failure for medical devices**. Using historical device data, the app classifies devices into risk categories and provides actionable suggestions to mitigate potential hazards.

This project demonstrates the integration of **data preprocessing, machine learning, and a Flask web interface**, deployed on [Render](https://medical-device-failure-precition.onrender.com/).

---

## Features

- Upload or select device data from dropdown options  
- Predict device risk class using a trained **Random Forest Classifier**  
- View detailed **description and suggestions** for each risk class  
- Interactive and user-friendly web interface  
- Fully deployed and accessible online  

---

## Tech Stack

- **Backend:** Python, Flask  
- **Machine Learning:** scikit-learn, pandas, numpy  
- **Data Visualization:** matplotlib, seaborn (for internal analysis)  
- **Deployment:** Render  
- **Data Storage:** Excel dataset (`final_cts.xlsx`)  

---

## Installation (For Local Development)

1. **Clone the repository


Create a virtual environment

python -m venv .venv


Activate the virtual environment

# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# Windows CMD
.\.venv\Scripts\activate.bat

# Mac/Linux
source .venv/bin/activate


Install dependencies

pip install -r requirements.txt


Run the Flask application

python app.py


Open your browser

Navigate to: http://127.0.0.1:5000/ to access the application.

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>

python -m venv .venv
Usage

On the home page, select device information from the dropdown menus:

Classification

Code

Implanted

Device name

Manufacturer name

Click Predict to generate the risk classification.

View the predicted class, description, and suggestion for the device.

Example risk classes:

Class 1: High risk — immediate action required

Class 2: Moderate risk — monitor closely

Class 3: Low risk — routine maintenance sufficient

How It Works

Data Preprocessing

Handles missing values and encodes categorical features.

Uses LabelEncoder and SimpleImputer from scikit-learn.

Model Training

Trains a RandomForestClassifier on historical device data.

Splits data into training and testing sets and evaluates accuracy.

Prediction

Takes new device information from the web form.

Encodes inputs and predicts risk class using the trained model.

Returns predicted class, description, and suggestions to the user.

Project Structure
medical_device_failure_prediction/
│
├── app.py                  # Flask web application
├── prediction.py           # ML model training and prediction logic
├── final_cts.xlsx          # Dataset of medical devices
├── templates/
│   ├── index.html          # Home page template
│   └── result.html         # Prediction result template
├── static/                 # CSS, images, JS files (if any)
├── tests/                  # Optional unit tests
├── requirements.txt        # Python dependencies
└── README.md

🔗 Live Application

Medical Device Failure Prediction on Render
