import os
import glob
import base64
import json
import glob
from flask import *
from final_AI import train,test
import flask

# print os.getcwd()
dirname = os.path.dirname(__file__)
print (os.getcwd())
print (dirname)
print (os.path.join(os.getcwd(), 'test'))
print (__file__)

app = Flask(__name__)

def start_app():
		app.run(host="localhost", port=7777, debug=True)

@app.route("/")
def hello_world():
    resp=flask.Response("Welcome to AI Doctor")
    resp.headers['ngrok-skip-browser-warning'] ='Ignore'
    return resp

@app.route("/train") 
def RouteTrain():
    train()
    return "Training is Done"
@app.route("/test") 
def RouteTest():
    test()
    return "Testing is Done"

@app.route('/api/uploader', methods = ['GET', 'POST'])
def api_upload_file():
    if request.method == 'POST':
        img_base64 = request.values['img']
        img_base64 = bytes(img_base64, encoding='utf-8')
        globing(os.path.join(dirname, 'static/test'))
        f = open(os.path.join(dirname, 'static/test/Healthy_unknown.jpg'), "a+")
        with open(os.path.join(dirname, 'static/test/Healthy_unknown.jpg'), "wb") as fh:
            fh.write(base64.decodebytes(img_base64))
    #   f = open(os.path.join(dirname, 'static/test/Healthy_unknown_1.jpg'), "a+")
    #   with open(os.path.join(dirname, 'static/test/Healthy_unknown_1.jpg'), "wb") as fh:
    #     fh.write(base64.decodebytes(img_base64))
    #   f = open(os.path.join(dirname, 'static/test/Healthy_unknown_300.jpg'), "a+")
    #   with open(os.path.join(dirname, 'static/test/Healthy_unknown_300.jpg'), "wb") as fh:
    #     fh.write(base64.decodebytes(img_base64))
   
    result = test()
    data_rx={"Result":f"{result[0]}","Accuracy_Smaller":f"{result[1]}","Accuracy_Bigger":f"{result[2]}"}

    return json.dumps(data_rx)



def globing(path):
    files = glob.glob(path+'/*')
    for f in files:
        os.remove(f)

start_app()