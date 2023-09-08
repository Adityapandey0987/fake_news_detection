from flask import Flask,escape,request
from flask.templating import render_template
import pickle

vectorizer = pickle.load(open("vectorizer.pkl",'rb'))
model = pickle.load(open("finalised model.pkl",'rb'))

app = Flask(__name__)

@app.route("/")
def home():
  return render_template("index.html")

@app.route("/predict",methods = ["GET","POST"])
def prediction():
  if request.method == "POST":
    news = str(request.form['news'])
    predict = model.predict(vectorizer.transform([news]))
    print(predict)
    if(predict == 1):
      predictt = 'fake'
    else:
       predictt = 'real'
    return render_template("prediction.html",prediction_text = 'News Headline is {}'.format(predictt))
  else:
    return  render_template("prediction.html")
@app.route("/dataset")
def datsett():
  return render_template("indexdataset.html")
@app.route("/aboutus")
def aboutus():
  return render_template("aboutus.html")
@app.route("/about-project")
def project():
  return render_template("aboutproject.html")

if(__name__ == "__main__"):
    app.run()