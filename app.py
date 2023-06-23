from flask import Flask, request, jsonify
import traceback
import pickle
import re
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
from flask_cors import CORS

stop_words=['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 
'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]

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
    text = [ps.stem(word) for word in text if not word in stop_words]
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
