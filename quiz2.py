import streamlit as st
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load the pre-trained model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

# Suppress warning about weights not being initialized
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2, state_dict=model.state_dict() if not isinstance(model, type(model)) else None)

# Define the prediction function
def predict(text):
    # If a single example is provided, convert it to a list
    if isinstance(text, str):
        text = [text]

    # Encode the text into tokens
    encoded_text = tokenizer(text, padding='max_length', truncation=True, max_length=512, return_tensors='pt')
    input_ids = encoded_text['input_ids']
    attention_mask = encoded_text['attention_mask']

    # Run the text through the model
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits

    # Get the probability of hate speech
    hate_speech_probability = torch.softmax(logits, dim=1)[:, 1].tolist()

    # Determine the predictions
    predictions = ["Hate speech" if prob > 0.5 else "Not hate speech" for prob in hate_speech_probability]

    return predictions[0] if len(predictions) == 1 else predictions

# Custom CSS styles
custom_css = """
<style>
    .stTextInput {
        width: 100%;
        padding: 10px;
        border: 1px solid #ddd;
        border-radius: 5px;
        margin-top: 10px;
    }

    .styled-button {
        background-color: #4CAF50;
        color: white;
        padding: 10px 20px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        cursor: pointer;
        border-radius: 5px;
        margin-top: 10px;
    }

    .styled-button:hover {
        background-color: #45a049;
    }

    .stButton button {
        background-color: #4CAF50;
        color: white;
        padding: 10px 20px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        cursor: pointer;
        border-radius: 5px;
    }

    .stButton button:hover {
        background-color: #45a049;
    }

    .stRadio {
        padding: 10px;
        border: 1px solid #ddd;
        border-radius: 5px;
        margin-top: 10px;
    }
</style>
"""

# Inject custom CSS
st.markdown(custom_css, unsafe_allow_html=True)

# Create the Streamlit app with a navigation bar
st.title("Hate Speech Detector")

# Sidebar for navigation
nav_option = st.sidebar.radio("Navigation", ["Text Input", "CSV Upload"])

# Check the chosen navigation option
if nav_option == "Text Input":
    # Option to input text directly
    text_input = st.text_area("Enter your text here:")
    
    if st.button("Predict"):
        # If text is entered, use that for prediction
        if text_input:
            prediction = predict(text_input)
            st.subheader("Prediction:")
            st.write(prediction)
        else:
            st.warning("Please enter text before clicking 'Predict'.")

elif nav_option == "CSV Upload":
    # Option to upload a CSV file
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    
    if st.button("Predict"):
        # If a CSV file is uploaded, use the first column for prediction
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            if not df.empty and not df.columns.empty:
                text_column = df.columns[0]
                predictions = df[text_column].apply(predict)
                st.subheader("Predictions:")
                st.write(predictions)
            else:
                st.warning("The CSV file is empty or does not have a valid column.")
        else:
            st.warning("Please upload a CSV file before clicking 'Predict'.")
