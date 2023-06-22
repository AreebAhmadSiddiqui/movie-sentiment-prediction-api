from flask import Flask, request, jsonify
import traceback
import streamlit as st
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
from flask_cors import CORS
model1 = pickle.load(open('./models/MultinomialNB.pkl', 'rb'))
model2 = pickle.load(open('./models/Bernoulli.pkl', 'rb'))
model3 = pickle.load(open('./models/Gaussian.pkl', 'rb'))
model4 = pickle.load(open('./models/Multinomial.pkl', 'rb'))
CountVectorizer = pickle.load(open('./models/CountVectorizer.pkl', 'rb'))

def data_cleaning(text):
    corpus=[]
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    text = text.split()
    text = [ps.stem(word) for word in text if not word in stopwords.words('english')]
    text = ' '.join(text)
    corpus.append(text)
    return countVectorizer(corpus)


def prediction(vector):
    list=[model1.predict(vector)[0],model2.predict(vector)[0],model3.predict(vector)[0],model4.predict(vector)[0]]

    cntOne,cntZero=0,0
    for x in list:
        cntOne+=x==1
        cntZero+=x==0
    
    if cntOne>=cntZero:
        return 1
    else:
        return 0
    

def countVectorizer(text):
    vector = CountVectorizer.transform(text).toarray()
    return prediction(vector)


app = Flask(__name__)
CORS(app)

@app.route('/sentiment',methods=['POST'])
def predict():
    try:
        data = request.get_json()
        prediction=data_cleaning(data['review'])
        return jsonify({'prediction': prediction})

    except:
        return jsonify({'trace': traceback.format_exc()})

if __name__ == '__main__':

    app.run(host='0.0.0.0',port=5000,debug=True)
