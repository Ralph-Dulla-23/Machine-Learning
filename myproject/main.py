# app.py
from flask import Flask, request, render_template, jsonify
import pickle
import re

app = Flask(__name__)

# Load the saved model and vectorizer
with open('best_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = ' '.join(text.split())
    return text

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the text from the POST request
        text = request.json['text']
        # Preprocess the text
        processed_text = preprocess_text(text)
        # Transform text using vectorizer
        text_vector = vectorizer.transform([processed_text])
        # Make prediction
        prediction = model.predict(text_vector)[0]
        return jsonify({'emotion': prediction})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)