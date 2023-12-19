import pandas as pd

from flask import Flask, render_template, request
from yhuhsshs import calculate_sustainability, transform_text_data



app = Flask(__name__)

def transform_text_data(df):
    # Assuming user_input_text is the text entered by the user
    user_input_text = "Your product description here..."

    # Transform the user input text
    user_input_df = pd.DataFrame({'user_input': [user_input_text]})
    user_input_df = transform_text_data(user_input_df)
    return df

def calculate_sustainability(user_input_df):
    # Use the machine learning model to make a prediction
    prediction = calculate_sustainability(user_input_df)
    return prediction


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get user input from the form
    user_input_text = request.form['value_proposition']

    # Transform the user input text
    user_input_df = pd.DataFrame({'user_input': [user_input_text]})
    user_input_df = transform_text_data(user_input_df)

    # Use the machine learning model to make a prediction
    prediction = calculate_sustainability(user_input_df)

    # Render the result page with the calculated sustainability score
    return render_template('result.html', sustainability_score=prediction)

if __name__ == '__main__':
    app.run(debug=True)
