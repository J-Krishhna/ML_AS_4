## SCENARIO 1: SMS Spam Classification using Multinomial Naïve Bayes

This project classifies SMS messages as **Spam** or **Ham (Not Spam)** using text preprocessing, vectorization, and a Multinomial Naïve Bayes classifier.

## Problem Statement

Classify SMS messages as Spam or Ham based on text content.

## Dataset

**Source:** Kaggle – SMS Spam Collection Dataset  
https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset  

- Target Variable: Message Label (Spam / Ham)
- Input Feature: SMS Text Message

##  Objectives

- Load and inspect the dataset using Pandas  
- Perform text preprocessing (lowercase conversion, punctuation removal)  
- Convert text into numerical features using Count Vectorization  
- Encode target labels (Spam = 1, Ham = 0)  
- Split dataset into training and testing sets  
- Train a Multinomial Naïve Bayes classifier  
- Evaluate performance using Accuracy, Precision, Recall, and F1 Score  
- Apply Laplace smoothing  
- Visualize Confusion Matrix  
- Identify top spam-indicating words  

##  Model Performance

- Accuracy: ~97%
- Precision: ~0.87
- Recall: ~0.90
- F1 Score: ~0.89

## Key Insights

- Words like **free, call, txt, claim, mobile, ur, now** strongly indicate spam.
- Vectorization transforms text into numerical form enabling ML training.
- Laplace smoothing prevents zero-probability issues.
- Naïve Bayes performs well for text due to high-dimensional independent word features.


 ## SCENARIO 2: Iris Flower Species Classification using Gaussian Naïve Bayes

This project classifies iris flowers into three species — **Setosa, Versicolor, and Virginica** — using physical measurements and a Gaussian Naïve Bayes classifier.

##  Problem Statement

Classify flower species based on physical measurements:

- Sepal Length  
- Sepal Width  
- Petal Length  
- Petal Width
- 
## Dataset

**Source:** Scikit-learn – Iris Dataset (Built-in)

- Target Variable: Flower Species
- Input Features: Sepal & Petal measurements

---

## Objectives

- Load and inspect the Iris dataset  
- Perform preprocessing and data inspection  
- Apply feature scaling using StandardScaler  
- Split dataset into training and testing sets  
- Train a Gaussian Naïve Bayes classifier  
- Evaluate performance using Accuracy, Precision, Recall, and F1 Score  
- Analyze class probabilities  
- Compare with Logistic Regression  
- Visualize Confusion Matrix and Decision Boundary  

---

##  Model Performance

- Accuracy: 100%
- Precision: 1.0
- Recall: 1.0
- F1 Score: 1.0
(Logistic Regression also achieved 100% accuracy)

## Key Insights

- Gaussian NB is suitable for continuous numerical features.
- Feature scaling improves numerical stability.
- Class probabilities show high confidence due to clear class separation.
- Performance is comparable to Logistic Regression.
- Iris dataset is well-separated, enabling near-perfect classification.
