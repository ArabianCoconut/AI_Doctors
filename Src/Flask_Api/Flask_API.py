# flask  --app Flask_app --debug run --port 7777
from tracemalloc import start
from flask import *
import json
web = Flask(__name__)

def start_app():
    web.run(host="localhost",debug=True,port=7770)

@web.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@web.route("/data", methods=['Post'])
def data():
    data=request.form['name']
    url=url_for("patient_data", name=data)
    return redirect(url)

@web.route('/patient_data/<name>', methods=['GET'])
def patient_data(name):
    data_rx={"Patient":f"{escape(name)}","Age":f"{request.form['age']}"}
    return json.dumps(data_rx)

## Start the app
start_app()