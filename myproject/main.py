import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from flask import Flask, request, render_template

app = Flask(__name__)

# Get the absolute path to your current working directory
current_dir = os.getcwd()
csv_file = os.path.join(current_dir, 'breast_cancer_dataset.csv')

# Load and prepare the data
data = pd.read_csv(csv_file)
X = data.iloc[:, 1:-1].values
y = data.iloc[:, -1].values

# Split and scale the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Train the SVM model (best performer from your results)
model = SVC(random_state=42)
model.fit(X_train_scaled, y_train)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get values from the form
    features = ['clump_thickness', 'uniformity_cell_size', 'uniformity_cell_shape',
               'marginal_adhesion', 'single_epithelial_cell_size', 'bland_chromatin',
               'normal_nucleoli', 'mitoses']
    
    # Create input array from form data
    input_values = []
    for feature in features:
        value = float(request.form[feature])
        input_values.append(value)
    
    # Scale the input values
    input_scaled = scaler.transform([input_values])
    
    # Make prediction
    prediction = model.predict(input_scaled)[0]
    
    # Convert prediction to meaningful text
    result = "Malignant (Class 4)" if prediction == 4 else "Benign (Class 2)"
    
    # Get the classifier used
    classifier = request.form['classifier']
    
    # Format input values for display
    inputs_display = ", ".join([f"{feat}: {val}" for feat, val in zip(features, input_values)])
    
    return render_template('result.html',
                         inputs=inputs_display,
                         model=classifier,
                         prediction=result)

if __name__ == '__main__':
    app.run(debug=True)