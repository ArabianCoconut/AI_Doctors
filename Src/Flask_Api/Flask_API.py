# TO RUN: flask  --app Flask_app --debug run --port 7777
from flask import *
import json

from sklearn.utils import resample
web = Flask(__name__)
data_rec = {}
def start_app():
    web.run(host="localhost",debug=True,port=7777)

@web.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@web.route("/data", methods=['Post'])
def data():  # DATA CAPTURE insert all form capture data.
    global data_rec  ## Records form input in a dictionary
    data_rec = {"name":request.form['name'],"age":request.form['age']}
    url = url_for('patient_data', name=data_rec['name'])
    return redirect(url)

@web.route('/patient_data/<name>', methods=['GET'])
def patient_data(name):
    data_rx={"Patient":f"{escape(name)}","Age":f"{escape(data_rec['age'])}"}  ## Calls data from dictionary
    return json.dumps(data_rx)

## Start the app
start_app()
