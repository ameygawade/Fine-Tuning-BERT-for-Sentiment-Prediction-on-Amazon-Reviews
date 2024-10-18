# BERT Sentiment Classifier: Fine-Tuning for Amazon Reviews
This project focuses on fine-tuning a BERT-based Sentiment Classifier on Amazon reviews to predict the sentiment of product reviews as either positive or negative. The model uses a pre-trained bert-base-uncased transformer from Hugging Face's transformers library and was fine-tuned for binary classification.

## Project Overview

### Key Features:
- Dataset: 
    - Amazon product reviews containing verified reviews, ratings, and feedback (1 for positive, 0 for negative).
- Preprocessing: 
    - Handling of null and empty entries, non-ASCII characters, tokenization using the BERT tokenizer.
- Fine-Tuning: 
    - The BERT model is fine-tuned using labeled sentiment data.
- Handling Class Imbalance: 
    - Class weights are calculated to handle the imbalance in the dataset (positive reviews dominate the dataset).
- Evaluation: 
    - Model performance is evaluated using metrics like accuracy, F1 score, confusion matrix, precision, and recall.
- Deployment-Ready: 
    - The model and tokenizer can be saved for deployment in a production environment.

### Results:
    - Accuracy: 95.77%
    - F1 Score: 0.9775
    - Precision: 0.97 for positive class and 0.73 for negative class
    - Recall: 0.99 for positive class and 0.55 for negative class

# Installation and SetupInstallation and Setup
### Clone the repository:
    git clone https://github.com/ameygawade/Fine-Tuning-BERT-for-Sentiment-Prediction-on-Amazon-Reviews.git

### Navigate to project directory:
    cd Fine-Tuning-BERT-for-Sentiment-Prediction-on-Amazon-Reviews


