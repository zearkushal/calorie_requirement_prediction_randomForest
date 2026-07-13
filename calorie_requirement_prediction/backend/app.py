from flask import Flask, render_template, request
import joblib
import pandas as pd
import os

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, '..', 'model', 'calorie_requirement_prediction_RF.joblib')
# Load your saved model
model = joblib.load(MODEL_PATH)
# The EXACT 12 feature columns in order
FEATURE_COLUMNS = [
    'Age',
    'Gender',
    'Height_cm',
    'Weight_kg',
    'Activity_Level',
    'Water_Intake_Liters',
    'Health_Status',
    'Diet_Type_High Protein',
    'Diet_Type_Keto',
    'Diet_Type_Mediterranean',
    'Diet_Type_Vegan',
    'Diet_Type_Vegetarian'
]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # 1. Extract numerical and mapped fields from form
    age = int(request.form['Age'])
    gender = int(request.form['Gender'])
    height = float(request.form['Height_cm'])
    weight = float(request.form['Weight_kg'])
    activity = int(request.form['Activity_Level'])
    water = float(request.form['Water_Intake_Liters'])
    health = int(request.form['Health_Status'])
    
    # 2. Handle the One-Hot Encoded Diet_Type
    selected_diet = request.form['Diet_Type']
    
    # Set all diet dummies to 0 initially
    diet_high_protein = 1 if selected_diet == 'High Protein' else 0
    diet_keto = 1 if selected_diet == 'Keto' else 0
    diet_mediterranean = 1 if selected_diet == 'Mediterranean' else 0
    diet_vegan = 1 if selected_diet == 'Vegan' else 0
    diet_vegetarian = 1 if selected_diet == 'Vegetarian' else 0
    # Note: If they chose "Balanced", all 5 binary flags remain 0 (because drop_first=True during training)

    # 3. Build a dictionary with EXACT matching column names
    input_data = {
        'Age': age,
        'Gender': gender,
        'Height_cm': height,
        'Weight_kg': weight,
        'Activity_Level': activity,
        'Water_Intake_Liters': water,
        'Health_Status': health,
        'Diet_Type_High Protein': diet_high_protein,
        'Diet_Type_Keto': diet_keto,
        'Diet_Type_Mediterranean': diet_mediterranean,
        'Diet_Type_Vegan': diet_vegan,
        'Diet_Type_Vegetarian': diet_vegetarian
    }

    # 4. Convert to DataFrame and enforce column order
    input_df = pd.DataFrame([input_data])[FEATURE_COLUMNS]

    # 5. Predict
    prediction = model.predict(input_df)[0]

    return render_template('index.html', prediction=round(prediction, 2))

if __name__ == '__main__':
    app.run(debug=True)