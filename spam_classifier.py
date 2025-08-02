import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix

#load dataset from CSV file
data = pd.read_csv('spam.csv', encoding='ISO-8859-1')

# Keep only relevant columns
data = data[['v1', 'v2']]
data.columns = ['label', 'message']

#convert labels to binary
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# Drop any rows with missing messages
data = data.dropna(subset=['message'])

# Make sure all messages are strings
data['message'] = data['message'].astype(str)

#3 Text preprocessing (vectorization)
vectorizer = CountVectorizer()
x = vectorizer.fit_transform(data['message'])
y = data['label']

#Split into training and testing
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#Train classifier
model = MultinomialNB()
model.fit(X_train, y_train)

#Test the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print (f"Model Accuracy: {accuracy * 100:.2f}%")

# Try it on a custom message
def predict_message(msg):
    vector = vectorizer.transform([msg])
    prediction  = model.predict(vector)
    return "Spam" if prediction == 0 else "Not Spam"

#Example
test_msg = "Congratulations! You've won MK1000000, call 0912345678 to collect your money."
print(f"Test Message: '{test_msg}' Prediction: '{predict_message(test_msg)}'")