import os
import glob
from io import BytesIO
import base64
import json
import glob
from flask import *
import New_Ai
from New_Ai import train, test

# print os.getcwd()
dirname = os.path.dirname(__file__)
print (os.getcwd())
print (dirname)
print (os.path.join(os.getcwd(), 'test'))
print (__file__)

app = Flask(__name__)

def start_app():
		app.run(host="localhost", port=5000, debug=True)

@app.route("/")
def hello_world():
    return "Welcome to AI Doctor"

@app.route("/train") 
def RouteTrain():
    train()
    return "Training is Done"
@app.route("/test") 
def RouteTest   ():
    test()
    return "Testing is Done"

@app.route('/api/uploader', methods = ['GET', 'POST'])
def api_upload_file():

   if request.method == 'POST':

      img_base64 = request.form['img']
      img_base64 = bytes(img_base64, encoding='utf-8')
      globing(os.path.join(dirname, 'static/test'))
      f = open(os.path.join(dirname, 'static/test/unknown.jpg'), "a+")
      with open(os.path.join(dirname, 'static/test/unknown.jpg'), "wb") as fh:
        fh.write(base64.decodebytes(img_base64))

      result = test()
      data_rx={"Result":f"{result[0]}","Accuracy":f"{result[1]}"}

      return json.dumps(data_rx)



def globing(path):
    files = glob.glob(path+'/*')
    for f in files:
        os.remove(f)

start_app()