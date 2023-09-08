#importing the dependencies
import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score 

import nltk
nltk.download('stopwords')

#printing the stowords
print(stopwords.words('english'))

#data preprocessing
news_dataset = pd.read_csv('train.csv')
news_dataset.shape

#first 5 rows of dataset
news_dataset.head()

#counting the number of missing alues in dataset
news_dataset.isnull().sum()

#replacing the empty values to null strings
news_dataset = news_dataset.fillna('')

#merging theauther name and news title
news_dataset['content'] = news_dataset['author'] + " "+ news_dataset['title']

#separating the data and label
X = news_dataset.drop(columns = 'label',axis=1)
Y = news_dataset['label']

#stemming(removing the prefix and suffix to get the root word)
port_stem = PorterStemmer()

def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]',' ',content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english') ]
    stemmed_content = " ".join(stemmed_content)
    return stemmed_content

news_dataset['content'] = news_dataset['content'].apply(stemming)

#separating the data and label
X = news_dataset['content'].values
Y = news_dataset['label'].values

#converting the texual data to numerical data
vectorizer = TfidfVectorizer()
vectorizer.fit(X)

X = vectorizer.transform(X)

#splitting the dataset to training and test data
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=2)

#training the model
model = LogisticRegression()
model.fit(X_train,Y_train)

#accuracy score on training data
X_train_pred = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_pred,Y_train)

print("Accuracy value for training data : ",training_data_accuracy)

#accuracy score on test data
X_test_pred = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_pred,Y_test)

print("Accuracy value for test data : ",test_data_accuracy)

#making a predictive system
X_new = X_test[0]
prediction = model.predict(X_new)

print(prediction)

if prediction[0] == 0:
    print("The news is real")
else:
    print("The news is fake")

#save mode
import pickle
filename = 'finalised model.pkl'
pickle.dump(model,open(filename,'wb'))

filename = 'vectorizer.pkl'
pickle.dump(vectorizer,open(filename,'wb'))