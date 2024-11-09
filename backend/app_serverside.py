import streamlit as st
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

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

def update_database_with_changes(df):
    # Clear the table and reinsert updated data without specifying the 'id' column
    cursor.execute("DELETE FROM predictions")
    conn.commit()
    
    # Insert updated rows, excluding 'id' to allow auto-increment
    for _, row in df.iterrows():
        cursor.execute('''INSERT INTO predictions (Review, Sentiment_prediction, Prediction_is_correct) 
                          VALUES (?, ?, ?)''', 
                       (row['Review'], row['Sentiment_prediction'], row['Prediction_is_correct']))
    conn.commit()

def plot_prediction_correctness(df):
    # Plot the overall correctness of predictions
    correctness_counts = df['Prediction_is_correct'].value_counts()
    labels = ['Correct', 'Incorrect']
    plt.figure(figsize=(5, 5))
    plt.pie(correctness_counts, labels=labels, autopct='%1.1f%%', startangle=140)
    plt.title("Overall Prediction Correctness")
    st.pyplot(plt)

def prepare_data_for_analysis(df):
    # Create a copy of the DataFrame to add analysis columns without modifying the original
    analysis_df = df.copy()
    
    # Add Correct_sentiment column based on conditions
    analysis_df['Correct_sentiment'] = analysis_df.apply(
        lambda row: 1 if row['Sentiment_prediction'] == 'Positive' and row['Prediction_is_correct'] == 1 else
                    0 if row['Sentiment_prediction'] == 'Negative' and row['Prediction_is_correct'] == 1 else
                    1 if row['Sentiment_prediction'] == 'Negative' and row['Prediction_is_correct'] == 0 else
                    0,
        axis=1
    )
    
    # Add Sentiment_binary column for confusion matrix calculations
    analysis_df['Sentiment_binary'] = analysis_df['Sentiment_prediction'].apply(lambda x: 1 if x == "Positive" else 0)
    
    return analysis_df

def plot_confusion_matrix(df):
    # Prepare the DataFrame for the confusion matrix
    analysis_df = prepare_data_for_analysis(df)
    
    # Use Correct_sentiment as the true labels and Sentiment_binary as the predicted labels
    y_true = analysis_df['Correct_sentiment']
    y_pred = analysis_df['Sentiment_binary']
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    
    # Plot the confusion matrix
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title('Confusion Matrix of Predictions')
    st.pyplot(plt)

# Streamlit interface
st.set_page_config(layout="wide")
st.title("Dashboard for Model Performance Analysis")
st.write("Manage saved predictions in the database:")

# Load saved data into session state if not already there
if "saved_data" not in st.session_state:
    st.session_state.saved_data = view_saved_data()

# Data Analysis Section
st.header("Data Analysis")

col1, col2, col3 = st.columns(3)

with col1:
    with st.container(border=True):
        # Display the prediction correctness chart
        plot_prediction_correctness(st.session_state.saved_data)

with col2:
    with st.container(border=True):
        # Display the confusion matrix
        plot_confusion_matrix(st.session_state.saved_data)

with st.expander("Click here to check database in detail"):
    # Form for saving edits
    with st.form("data_editor_form"):
        # st.write(st.session_state.saved_data)
        # Display editable data in data editor
        edited_data = st.data_editor(st.session_state.saved_data, num_rows="dynamic")  
        
        # Update session state with any edits made
        st.session_state.saved_data = edited_data  # Update session state with edited DataFrame

        # Button to save changes to the database
        save_changes = st.form_submit_button("Save Changes")
        
        if save_changes:
            # Update the database with edited data
            update_database_with_changes(st.session_state.saved_data)
            st.success("Changes saved to the database.")