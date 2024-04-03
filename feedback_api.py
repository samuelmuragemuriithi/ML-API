from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import os
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from keras import regularizers
from sklearn.model_selection import cross_val_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from keras.utils import to_categorical


app = Flask(__name__)

# Section 1: Importing libraries and downloading NLTK resources
nltk.download('stopwords')

# Section 2: Data Injection
path1 = 'C:/Users/user/Downloads/OPINIONAI/Feedback-Analysis(ML Code)/Data/bugs.txt'
path2 = 'C:/Users/user/Downloads/OPINIONAI/Feedback-Analysis(ML Code)/Data/comments.txt'
path3 = 'C:/Users/user/Downloads/OPINIONAI/Feedback-Analysis(ML Code)/Data/complaints.txt'
path4 = 'C:/Users/user/Downloads/OPINIONAI/Feedback-Analysis(ML Code)/Data/meaningless.txt'
path5 = 'C:/Users/user/Downloads/OPINIONAI/Feedback-Analysis(ML Code)/Data/requests.txt'

# Section 3: Function to read text data from files
def text_data(path):
    text_Body = []
    with open(path, "r", encoding='windows-1256') as f:
        lines = f.readlines()
        text_Body.append(lines)
    text_body_appended = []
    for i in range(0,len(text_Body[0])):
        value = text_Body[0][i]
        text_body_appended.append(value)
    return text_body_appended

# Section 4: Read data from files
bugs = text_data(path1)
comments = text_data(path2)
complaints = text_data(path3)
meaningless = text_data(path4)
requests = text_data(path5)

# Section 5: Data Visualization
print(len(bugs))
print(len(comments))
print(len(complaints))
print(len(meaningless))
print(len(requests))

# Section 6: Function to create DataFrame
def data_frame(txt, category):
    column_names = ('text', 'Category')
    df = pd.DataFrame(columns=column_names)
    df['text'] = txt
    df['Category'] = category
    return df

# Section 7: Create DataFrame for each category
data = data_frame(bugs, "Bug")
data = pd.concat([data, data_frame(comments, "comments")])
data = pd.concat([data, data_frame(complaints, "complaints")])
data = pd.concat([data, data_frame(meaningless, "meaningless")])
data = pd.concat([data, data_frame(requests, "requests")])


# Section 8: Unique categories
data['Category'].unique()

# Section 9: Display DataFrame head
data.head()

# Section 10: Group by Category
data.groupby(['Category']).size()

# Section 11: Count plot
import matplotlib.pyplot as plt
import seaborn as sns
sns.countplot(x='Category',data=data,order=['Bug', 'comments', 'complaints', 'meaningless', 'requests'])
plt.show()

# Section 12: Data Cleaning and Preparation
data['text'] = data['text'].map(lambda x: re.sub('http.*', '', str(x)))  # Remove links
data['text'] = data['text'].map(lambda x: re.sub('[0-9]', '', str(x)))  # Remove numeric values
data['text'] = data['text'].map(lambda x: re.sub('[#|*|$|:|\\|&]', '', str(x)))  # Remove special characters

# Section 13: Define custom stopwords
my_stopwords = ['jan', 'january', 'february' 'feb', 'march', 'april', 'may', 'june', 'july', 'aug',
                'october', 'October', 'june', 'july', 'February', 'apr', 'Apr', 'february', 'jun', 'jul', 'feb', 'sep',
                'august', 'sept', 'september', 'oct', 'october', 'nov', 'november', 'dec', 'december', 'mar', 'november october', 'wasnt']

# Section 14: Prepare data for training
stop = stopwords.words('english')
text = []
none = data['text'].map(lambda x: text.append(' '.join([word for word in str(x).strip().split() if word not in stop and word not in my_stopwords])))
tfid = TfidfVectorizer(strip_accents=None, lowercase=False, preprocessor=None)
x_features = tfid.fit_transform(text).toarray()
x_features = pd.DataFrame(x_features)

# Section 15: One Vs All Training
target = data['Category']
label = LabelEncoder()
target = label.fit_transform(target)
target = to_categorical(target)
target = pd.DataFrame(data=target, columns=['Bug', 'comments', 'complaints', 'meaningless', 'requests'])

# Section 16: Define Logistic Regression model
logistic = LogisticRegression(penalty='l2', solver='newton-cg', C=5, multi_class='ovr', max_iter=5000)

# Section 17: Train Bugs vs. All
acc = cross_val_score(estimator=logistic, X=x_features.iloc[:144, :], y=target.iloc[:144, 0], cv=5)
acc.mean()

# Section 18: Train Comments vs. All
acc = cross_val_score(estimator=logistic, X=x_features.iloc[100:2100, :], y=target.iloc[100:2100, 1], cv=5)
acc.mean()

# Section 19: Train Complaints vs. All
acc = cross_val_score(estimator=logistic, X=x_features.iloc[1500:3500, :], y=target.iloc[1500:3500, 2], cv=5)
acc.mean()

# Section 20: Train Meaningless vs. All
acc = cross_val_score(estimator=logistic, X=x_features.iloc[2700:3300, :], y=target.iloc[2700:3300, 3], cv=5)
acc.mean()

# Section 21: Train Requests vs. All
acc = cross_val_score(estimator=logistic, X=x_features.iloc[-206:, :], y=target.iloc[-206:, 4], cv=5)
acc.mean()

# Section 22: Define ANN model
clf = Sequential()
clf.add(Dense(units=2048, activation="relu", kernel_initializer="uniform", kernel_regularizer=regularizers.l2(0.001), input_dim=5193))
clf.add(Dropout(0.2))
clf.add(Dense(units=2048, activation="relu", kernel_initializer="uniform", kernel_regularizer=regularizers.l2(0.001)))
clf.add(Dropout(0.2))
clf.add(Dense(units=2048, activation="relu", kernel_initializer="uniform", kernel_regularizer=regularizers.l2(0.001)))
clf.add(Dropout(0.2))
clf.add(Dense(units=5, activation="softmax", kernel_initializer="uniform"))
clf.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Section 23: Train ANN
hist = clf.fit(x_features, target, batch_size=32, epochs=24)

# Section 24: Evaluate ANN by Graph
fig, ax = plt.subplots(2,1)
ax[0].plot(hist.history['loss'], color='b', label="Training loss")
legend = ax[0].legend(loc='best', shadow=True)

if 'accuracy' in hist.history:
    ax[1].plot(hist.history['accuracy'], color='r', label="Training accuracy")
    legend = ax[1].legend(loc='best', shadow=True)
else:
    print("Accuracy information not found in history.")

# Section 40: Save the trained model
clf.save('C:/Users/user/feedback_clfr.keras')

# Section 33: Define the predict route
@app.route('/predict', methods=['POST'])
def predict():
    # Get data from the request
    data = request.get_json()
    test_data = data.get('test')  # Extract the 'test' data from the request

    # Convert the test data to a DataFrame
    test_df = pd.DataFrame(test_data, columns=['text'])

    # Preprocess the input data
    test_df['text'] = test_df['text'].map(lambda x: re.sub('http.*', '', str(x)))  # Remove links starting with https
    test_df['text'] = test_df['text'].map(lambda x: re.sub('[0-9]', '', str(x)))  # Remove numeric values
    test_df['text'] = test_df['text'].map(lambda x: re.sub('[#|*|$|:|\\|&]', '', str(x)))  # Remove special characters

    # Remove custom stopwords and tokenize
    stop = stopwords.words('english')
    text = []
    none = test_df['text'].map(lambda x: text.append(' '.join([word for word in str(x).strip().split() if word.lower() not in stop])))
    
    # Vectorize the text data using TF-IDF
    tfid = TfidfVectorizer(strip_accents=None, lowercase=False, preprocessor=None)
    x_features_test = tfid.fit_transform(text).toarray()
    x_features_test = pd.DataFrame(x_features_test)

    # Predict using the trained model
    results = clf.predict(x_features_test)

    # Select the index with the maximum probability
    results = np.argmax(results, axis=1)

    # Convert predicted indices to category labels
    int_category = {0: 'Bug', 1: 'comments', 2: 'complaints', 3: 'meaningless', 4: 'requests'}
    results = pd.DataFrame(results, columns=['Category'])
    results['Category'] = results['Category'].apply(lambda x: int_category[x])

    # Merge predictions with the original text data
    results['text'] = test_df['text']

    # Return the predictions as a JSON response
    return jsonify({'predictions': results.to_dict(orient='records')})  # Convert DataFrame to dictionary for JSON serialization

# Section 34: Start the Flask application
if __name__ == '__main__':
    # Check if the trained model file exists
    if os.path.exists('C:/Users/user/feedback_clfr.keras'):
        # Load the trained model
        clf = load_model('C:/Users/user/feedback_clfr.keras')
    else:
        # Train the model if the file doesn't exist
        hist = clf.fit(x_features, target, batch_size=32, epochs=24)
        clf.save('feedback_clfr.h5')
    
    app.run(debug=True)
