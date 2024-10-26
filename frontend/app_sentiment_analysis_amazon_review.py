import streamlit as st

# Streamlit interface
st.title("Sentiment Analysis: Amazon Review Classifier")
st.write("Enter a product review to get the predicted sentiment:")


# Input text from user
user_input = st.text_area("Enter your review here", max_chars=5000)

if st.button("Submit review"):
    if user_input:
        sentiment = ("Positive")
        st.write(f"the predicted sentiment is: {sentiment}")
        prediction_status = st.radio("Is predicted sentiment correct?",
                                     ["Yes","No"])
        if prediction_status == "No":
            correct_sentiment = st.radio("Select the correct setiment",
                     ["Positive","Negative"])
            if correct_sentiment == sentiment:
                st.write("the predicted sentiment and corrected sentiment is same")
            else:
                sentiment_type = st.text_input("Plese mention type of review you submitted","For example for Sarcastic, Positive review with negative emoji and etc.")
                if st.button("Submit the correction"):
                    st.write("Correction submitted")
        else:
            if st.button("Submit the review"):
                st.write("Review submitted")
        
    else:
        st.write("Kindly mention your review")


