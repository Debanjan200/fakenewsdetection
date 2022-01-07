from flask import Flask,render_template,request
import nltk
from nltk.corpus import stopwords
import numpy as np
import re
from nltk.stem.porter import PorterStemmer
import pickle

nltk.download("stopwords")
cv=pickle.load(open("transform.pkl","rb"))
model=pickle.load(open("transfer.pkl","rb"))
encoder=pickle.load(open("encoder.pkl","rb"))
app=Flask(__name__)

def prediction(text):
    test_corpus=[]
    
    sentiment=re.sub("[^a-zA-Z]",' ',text)
    sentiment=sentiment.lower()
    sentiment=sentiment.split()
    ps=PorterStemmer()
    all_stopwords=stopwords.words("english")
    all_stopwords.remove("not")
    sentiment=[ps.stem(word) for word in sentiment if not word in set(all_stopwords)]
    sentiment=' '.join(sentiment)
    test_corpus.append(sentiment)
    test=cv.transform(test_corpus).toarray()
    print(test.shape)
    pred=encoder.inverse_transform([np.argmax(model.predict(test))])

    return pred[0]


@app.route("/")
def home():
    return render_template("index.html")

@app.route("/",methods=["POST"])
def predict():
    if request.method=="POST":
        text=(request.form["sentiment"])

    predict=prediction(text)

    return render_template("index.html",prediction=predict)

if __name__=="__main__":
    app.run(debug=True)