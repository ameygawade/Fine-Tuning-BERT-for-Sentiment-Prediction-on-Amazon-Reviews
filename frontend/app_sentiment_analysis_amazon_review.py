import streamlit as st
import pandas as pd
from transformers import BertForSequenceClassification, BertTokenizer
import torch
import emoji
import sqlite3

# Load the pre-trained BERT model and tokenizer
model = BertForSequenceClassification.from_pretrained('./bert_sentiment_v2')
model.eval()
tokenizer = BertTokenizer.from_pretrained('./bert_sentiment_tokenizer_v2')


# Set up SQLite database connection
conn = sqlite3.connect("sentiment_analysis.db", check_same_thread=False)
cursor = conn.cursor()

# Create a table for storing predictions
cursor.execute('''CREATE TABLE IF NOT EXISTS predictions 
                  (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                   Review TEXT, 
                   Sentiment_prediction TEXT, 
                   Prediction_is_correct BOOLEAN)''')
conn.commit()


# Streamlit interface
st.title("Sentiment Analysis")
st.write("Enter a product review to get the predicted sentiment:")

def convert_emoji_to_text(text):
    if isinstance(text, str):  # Only process if the input is a string
        return emoji.demojize(text)
    return text  # Return the original value if it's not a string

# Prediction function
def predict_sentiment(review):
    review = convert_emoji_to_text(review)
    inputs = tokenizer(review, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    sentiment = "Positive" if predicted_class == 1 else "Negative"
    return sentiment

def display_prediction(data):
    # Process each review and predict sentiment
    data["Sentiment_prediction"] = data["Review"].apply(predict_sentiment)
    data["Prediction_is_correct?"] = True  # Placeholder for feedback
    return data

# Function to save DataFrame to database
def save_to_database(df):
    for _, row in df.iterrows():
        cursor.execute('''INSERT INTO predictions (Review, Sentiment_prediction, Prediction_is_correct) 
                          VALUES (?, ?, ?)''', 
                       (row['Review'], row['Sentiment_prediction'], row['Prediction_is_correct?']))
    conn.commit()

# Input section with form
with st.form("Review"):
    user_input = st.text_area("Enter your review here", max_chars=5000)
    file_uploader = st.file_uploader(label="or upload your review file (.csv)", type="csv")
    submit_but = st.form_submit_button("Submit")

    if submit_but:
        st.header("Result")

        # Case 1: If a file is uploaded, process the file
        if file_uploader is not None:
            # Read CSV file
            data = pd.read_csv(file_uploader)
            
            # Ensure there's a "Review" column in the file
            if "Review" in data.columns:
                df = display_prediction(data)  # Predict sentiment for each review
                show_df = st.data_editor(df)  # Display editable DataFrame
            else:
                st.error("The uploaded CSV file must contain a 'Review' column.")
        
        # Case 2: If no file is uploaded, use the text input
        elif user_input:
            # Single review prediction
            df = pd.DataFrame([{"Review": user_input, "Sentiment_prediction": predict_sentiment(user_input), "Prediction_is_correct?": True}])
            show_df = st.data_editor(df)

with st.form("your prediction"):
    # Final submission for feedback
    submit_prediction = st.form_submit_button("Submit Prediction Result")
    if submit_prediction:
        save_to_database(df)
        st.balloons()
        st.write("Thank you for your input!")        
