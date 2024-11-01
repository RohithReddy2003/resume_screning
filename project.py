import streamlit as st 
import numpy as np 
import pandas as pd 
from sklearn.preprocessing import LabelEncoder 
from sklearn.model_selection import train_test_split 
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.multiclass import OneVsRestClassifier 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn import metrics 
import nltk 
import re 
from nltk.corpus import stopwords 
import string 
 
# Function to clean text 
def clean(text): 
    text = re.sub('http\S+\s*', ' ', text) 
    text = re.sub('RT|cc', ' ', text) 
    text = re.sub('#\S+', '', text) 
    text = re.sub('@\S+', '', text) 
    text = re.sub('[%s]' % re.escape("""!"#$%&'()*=,-./:;<=>?@[\]^_`{|}~"""), ' ', text) 
    text = re.sub('\s+', ' ', text) 
    text = re.sub(r'[^\x00-\x7f]', r' ', text) 
    return text 
 
# Main function 
def main(): 
    st.title("Resume Category Prediction") 
    st.markdown( 
    """ 
    <style> 
    .reportview-container { 
        background: url(r"C:/Users/hp/Downloads/resume.jpeg") no-repeat center center; 
        background-size: cover; 
    } 
    </style> 
    """, 
    unsafe_allow_html=True 
) 
 
 
    # File uploader 
    uploaded_file = st.file_uploader("Upload CSV file", type="csv") 
    if uploaded_file is not None: 
        data = pd.read_csv(uploaded_file) 
 
        # Clean text 
        data['clean text'] = data.Resume.apply(lambda x: clean(x)) 
 
        # Model training 
        var = ['Category'] 
        le = LabelEncoder() 
        for i in var: 
            data[i] = le.fit_transform(data[i]) 
 
        text = data['clean text'].values 
        target = data['Category'].values 
 
        vect = TfidfVectorizer( 
            sublinear_tf=True, 
            stop_words='english', 
            max_features=2000) 
 
        vect.fit(text) 
        Word_feature = vect.transform(text) 
 
        x_train, x_test, y_train, y_test = train_test_split(Word_feature, target, random_state=0, test_size=0.2) 
 
        model = OneVsRestClassifier(KNeighborsClassifier()) 
        model.fit(x_train, y_train) 
 
        prediction = model.predict(x_test) 
        training_score = model.score(x_train, y_train) 
        testing_score = model.score(x_test, y_test) 
 
        st.write("Training Accuracy: {:.2f}".format(training_score)) 
        st.write("Testing Accuracy: {:.2f}".format(testing_score)) 
 
main()