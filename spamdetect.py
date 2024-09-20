import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import nltk
from nltk.corpus import stopwords
import string


nltk.download('stopwords')

# Sample dataset 
data = {
    'message': [
        'Congratulations! You have won a lottery. Call now!',
        'Hey, are we still on for the meeting today?',
        'Free entry in a contest. Claim your prize now.',
        'Can we reschedule the appointment?',
        'You are selected for a special offer. Act now!',
        'I will be late for the call, sorry.'
    ],
    'label': ['spam', 'ham', 'spam', 'ham', 'spam', 'ham']
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Preprocessing function to clean the text data
def preprocess_message(message):
    message = message.lower()  # Lowercase
    message = ''.join([char for char in message if char not in string.punctuation])  # Remove punctuation
    message = ' '.join([word for word in message.split() if word not in stopwords.words('english')])  # Remove stopwords
    return message

# Apply preprocessing
df['message'] = df['message'].apply(preprocess_message)

# Split the data into train and test sets
X = df['message']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Convert text to feature vectors using CountVectorizer 
vectorizer = CountVectorizer()
X_train_counts = vectorizer.fit_transform(X_train)
X_test_counts = vectorizer.transform(X_test)

# Train a Naive Bayes classifier
model = MultinomialNB()
model.fit(X_train_counts, y_train)

# Make predictions
y_pred = model.predict(X_test_counts)

# Evaluate the model
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
print('Classification Report:')
print(classification_report(y_test, y_pred))

# Test with a custom message
def predict_spam(message):
    message = preprocess_message(message)
    message_vec = vectorizer.transform([message])
    prediction = model.predict(message_vec)
    return prediction[0]

# Example usage:
test_message = "Congratulations! You've won a free cruise to the Bahamas!"
print(f'Test message: "{test_message}" is classified as: {predict_spam(test_message)}')
