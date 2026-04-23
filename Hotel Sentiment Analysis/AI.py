import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from nltk.corpus import stopwords
import string
import nltk

# Download necessary NLTK resources
nltk.download('stopwords')

# Load the dataset
data_set = pd.read_csv('D:/tripadvisor_hotel_reviews.csv')

# Display the first few rows
print(data_set.head())

# Check for missing values
missing_values = data_set.isnull().sum()

missing_values = data_set['Rating'].isnull().sum()

if missing_values > 0:
    # Get the distribution of ratings 
    rating_counts = data_set['Rating'].value_counts()
    
    # If one value is significantly more frequent 
    if rating_counts.iloc[0] / rating_counts.sum() > 0.5:  # more than 50% of one value
        print("Ratings are highly concentrated around one value. Using mode for imputation.")
        data_set['Rating'] = data_set['Rating'].fillna(rating_counts.idxmax())  # Use the mode
    else:
        # Check for skewness
        skewness = data_set['Rating'].skew()
        
        # If the data is skewed, use median, else use mean
        if abs(skewness) > 1: 
            print("Ratings are skewed. Using median for imputation.")
            data_set['Rating'] = data_set['Rating'].fillna(data_set['Rating'].median())  
        else:
            print("Ratings are fairly symmetric. Using mean for imputation.")
            data_set['Rating'] = data_set['Rating'].fillna(data_set['Rating'].mean())  

    print(f"Missing values in 'Rating' have been filled.\n Updated dataset:\n{data_set.head()}")

# Visualize the rating distribution
sns.countplot(data=data_set, x='Rating')
plt.title("Distribution of Ratings")
plt.show()

# Label encoding for sentiments
def label_encode(x):
    if x == 1 or x == 2:
        return 0  # Negative
    if x == 3:
        return 1  # Neutral
    if x == 4 or x == 5:
        return 2  # Positive

data_set['sentiment'] = data_set['Rating'].apply(label_encode)

# Preprocess the reviews
def preprocess_text(text):
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    # Remove digits
    text = re.sub(r'\d+', '', text)
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    text = ' '.join(word for word in text.split() if word.lower() not in stop_words)
    
    return text

data_set['Review'] = data_set['Review'].apply(preprocess_text)

# Features and target variable
X = data_set['Review']
y = data_set['sentiment']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Vectorize the text data using TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 1))
X_train_tf = tfidf_vectorizer.fit_transform(X_train)
X_test_tf = tfidf_vectorizer.transform(X_test)

# Train Random Forest classifier
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train_tf, y_train)

# Predict and evaluate
y_pred = rf.predict(X_test_tf)
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Optional: Visualize feature importance
feature_importances = rf.feature_importances_
vocab = tfidf_vectorizer.get_feature_names_out()
importance_df = pd.DataFrame({'Feature': vocab, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False).head(10)

plt.figure(figsize=(10, 6))
sns.barplot(data=importance_df, x='Importance', y='Feature')
plt.title("Top 10 Important Features")
plt.show()
