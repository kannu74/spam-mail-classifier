#imports-->
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords  #for removing stopwords
from nltk.stem import PorterStemmer #for stemming
from nltk.tokenize import word_tokenize #for tokenizing sentence into words
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score,classification_report
import pickle


#Loading dataset
data=pd.read_csv("spam.csv")
data=data[['v1','v2']]
data.columns=['label','message']


#encoding labels
data['label']=data['label'].map({'ham':0,'spam':1})
# print(data.head())

#preprocessing data
nltk.download("punkt")
nltk.download("stopwords")

ps=PorterStemmer()
stop_words=set(stopwords.words("english"))

def preprocess_text(text):
    # Tokenize and lowercase
    words = word_tokenize(text.lower())
    # Remove stopwords and stem words
    filtered_words = [ps.stem(word) for word in words if word.isalnum() and word not in stop_words]
    return ' '.join(filtered_words)

data['message'] = data['message'].apply(preprocess_text)
print(data.head())

#Feature Extraction
vectorizer = TfidfVectorizer(max_features=3000)
X = vectorizer.fit_transform(data['message']).toarray()
y = data['label']

print("Feature Matrix Shape:", X.shape)

#Train Test Splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# print("Training Data Size:", X_train.shape)
# print("Testing Data Size:", X_test.shape)

model=MultinomialNB()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))



# Save the trained model
with open("spam_classifier_model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)

# Save the vectorizer
with open("vectorizer.pkl", "wb") as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)
