from flask import Flask, render_template, request, jsonify
import random
import json
import numpy as np
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
import pickle
import sklearn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load intents data
with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

# Load the neural network model
FILE = "data.pth"
data = torch.load(FILE, map_location=torch.device('cpu'))

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

app = Flask(__name__)

# Define questions for heart disease assessment
heart_disease_questions = [
    "How old are you?",
    "What is your gender?",
    "Please specify the type of chest pain. (0,1,2,3)",
    "What is your resting blood pressure? (90-200)",
    "What is your serum cholesterol level? (120-600)",
    "Is your fasting blood sugar level greater than 120mg/dl? (Yes/No)",
    "What is your resting electrocardiographic result? (0,2)",
    "What is your maximum heart rate achieved? (70-210)",
    "Do you experience exercise-induced angina? (Yes/No)",
    "What is your ST depression induced by exercise relative to rest? (0-6.5)",
    "What is the slope of the peak exercise ST segment? (0-2)",
    "How many major vessels (0-3) colored by fluoroscopy do you have?",
    "What is your thalassemia type? (0 = normal; 1 = fixed defect; 2 = reversible defect)"
]

# Initialize variables to keep track of heart disease assessment state
heart_disease_assessment = False
current_question_index = 0
user_responses = {}

# Function to reset heart disease assessment state
def reset_heart_disease_assessment():
    global heart_disease_assessment, current_question_index, user_responses
    heart_disease_assessment = False
    current_question_index = 0
    user_responses = {}

# Chatbot setup and initialization code goes here...

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/get", methods=["POST"])
def get_response():
    global heart_disease_assessment, current_question_index

    user_message = request.form['msg']
    response = ""

    # Check if user mentions medical or heart disease
    if "medical" in user_message.lower() or "heart disease" in user_message.lower():
        heart_disease_assessment = True
        current_question_index = 0
        response = heart_disease_questions[current_question_index]
    elif heart_disease_assessment:
        # If heart disease assessment is ongoing, collect user responses to questions
        if current_question_index < len(heart_disease_questions):
            user_responses[heart_disease_questions[current_question_index]] = user_message
            current_question_index += 1
            if current_question_index < len(heart_disease_questions):
                response = heart_disease_questions[current_question_index]
            else:
                # Heart disease assessment complete, make prediction
                response = make_heart_disease_prediction()
    else:
        # Process user message using the model
        response = process_user_message(user_message)

    return jsonify({'msg': response})

def process_user_message(message):
    sentence = tokenize(message)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    # Check if the predicted intent is a greeting
    
    # If the probability is high enough, respond based on the intent
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                return random.choice(intent['responses'])

    # If the probability is low or the intent is not recognized, provide a default response
    return "I do not understand..."


def convert_response_to_numeric(question, response):
    if "gender" in question.lower():
        return 1 if response.lower() == "male" else 0
    elif "blood sugar" in question.lower():
        return 1 if response.lower() == "yes" else 0
    elif "angina" in question.lower():
        return 1 if response.lower() == "yes" else 0
    else:
        try:
            return float(response)
        except ValueError:
            return None

def make_heart_disease_prediction():
    # Convert user responses to a feature vector
    feature_vector = []
    for question in heart_disease_questions:
        response = user_responses.get(question, "")
        if response:
            numeric_response = convert_response_to_numeric(question, response)
            if numeric_response is not None:
                feature_vector.append(numeric_response)
        else:
            feature_vector.append("")

    # Perform prediction using the heart disease model
    with open('heartdisease_model.pkl', 'rb') as file:
        heart_disease_model = pickle.load(file)

    user_input_numeric = np.array(feature_vector, dtype=float)
    prediction = heart_disease_model.predict(user_input_numeric.reshape(1, -1))

    reset_heart_disease_assessment()

    if prediction == 1:
        return "Our model predicts that you might have a heart disease."
    else:
        return "Our model predicts that you do not have a heart disease."


if __name__ == "__main__":
    app.run(debug=True)
