print("Jaya Krishna G - 24BAD042")
import pandas as pd
import numpy as np
import re
import string
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

df = pd.read_csv('spam.csv', encoding='latin-1')
df = df[['v1','v2']]
df.columns = ['label','message']

def clean_text(text):
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)
    text = re.sub(r"\d+", "", text)
    return text

df['cleaned'] = df['message'].apply(clean_text)

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['cleaned'])

encoder = LabelEncoder()
y = encoder.fit_transform(df['label'])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = MultinomialNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=encoder.classes_)
disp.plot()
plt.show()

feature_names = vectorizer.get_feature_names_out()
spam_probs = model.feature_log_prob_[1]
top_indices = np.argsort(spam_probs)[-20:]
top_words = [feature_names[i] for i in top_indices]
print("Top Spam Words:", top_words)

model_smooth = MultinomialNB(alpha=0.5)
model_smooth.fit(X_train, y_train)
y_pred_smooth = model_smooth.predict(X_test)
print("Accuracy with smoothing:", accuracy_score(y_test, y_pred_smooth))