import streamlit as st
import sqlite3
import pandas as pd

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

def view_saved_data():
    # Query to select all data from the predictions table
    query = "SELECT * FROM predictions"
    data = pd.read_sql(query, conn)
    return data

# Streamlit interface
st.title("Sentiment Analysis")
st.write("Enter a product review to get the predicted sentiment:")

# Add a section to display saved data
if st.button("Show All Saved Predictions"):
    saved_data = view_saved_data()  # Fetch data from the database
    if not saved_data.empty:
        st.write("Saved Predictions in Database:")
        st.data_editor(saved_data)  # Display the data in Streamlit
    else:
        st.write("No data saved in the database yet.")