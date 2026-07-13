# Daily Calorie Requirement Predictor 🥗🔥

A machine learning web application built with **Flask** and **Scikit-Learn** that predicts daily caloric intake requirements based on biometric metrics, activity levels, health status, and diet types.

---

## 📌 Project Overview

This application deploys a trained **Random Forest Regressor** model via a RESTful Flask server. The system accepts user physical parameters, automatically applies necessary feature transformations, and calculates an estimated daily energy requirement ($kcal/day$).

---

## 🛠️ Repository Structure

```text
healthy_diet_calorie_intake_RandomForest/
├── model/
│   └── calorie_requirement_prediction_RF.joblib   # Trained Model
├── calorie_requirement_prediction/
│   └── backend/
│       ├── app.py                                # Flask backend entry point
│       └── templates/
│           └── index.html                        # Web UI template
├── .gitignore                                    # Production exclusion rules
└── README.md                                     # Project documentation
