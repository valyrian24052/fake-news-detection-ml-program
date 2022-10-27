import csv
import numpy as np
import pandas as pd
import re
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from  sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import nltk 
# nltk.download('stopwords')

#dataset import 
dataset = pd.read_csv(r'C:\Users\Valyr\OneDrive\Desktop\projects\fake news\train.csv')
# print(dataset.shape)
dataset=dataset.fillna('')
dataset.isnull().sum()

dataset['content']=dataset['author']+' '+dataset['title']

# stemming words

stemmer=PorterStemmer()

def stemming(content):
    stmcom=re.sub('[^a-zA-Z]',' ',content)
    stmcom=stmcom.lower()
    stmcom=stmcom.split()
    stmcom=[stemmer.stem(word) for word in stmcom if not word in stopwords.words('english')]
    stmcom=' '.join(stmcom)
    return stmcom

dataset['content']=dataset['content'].apply(stemming)
# print(dataset['content'])

x=dataset['content'].values
y=dataset['label'].values

# print(x,y)

# converting the textual data to numerical data
vectorizer = TfidfVectorizer()
vectorizer.fit(x)

x=vectorizer.transform(x)
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = 0.2, stratify=y, random_state=2)

model=LogisticRegression()
model.fit(X_train,Y_train)

# accuracy score on the test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

print('Accuracy score of the test data : ', test_data_accuracy)

"""Making a Predictive System"""

X_new = X_test[3]

prediction = model.predict(X_new)
print(prediction)

if (prediction[0]==0):
  print('The news is Real')
else:
  print('The news is Fake')

print(Y_test[3])







