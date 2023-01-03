import argparse
import traceback
import os
import sys
import json
import tempfile

from flask import (
    Flask, 
    render_template, 
    jsonify,
    request, 
    Response, 
    url_for, 
)

app = Flask(__name__,
    static_url_path='', 
    static_folder='static',
    template_folder='templates',
)

@app.route("/ping")
def ping():
    return jsonify(success=True)

@app.route("/")
def serve():
    return render_template('home.html')

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
NOSTR_JSON_FILE = os.path.join(THIS_DIR,'nostr.json')
with open(NOSTR_JSON_FILE,'r') as f:
    nostr_dict = json.loads(f.read())

@app.route("/.well-known/nostr.json")
def well_known_nostr():
    try:
        if request.method == "GET":
            name = request.args.get("name")
            if name is not None and name == "aigonewrong":
                response = jsonify(nostr_dict)
                response.headers.add('Access-Control-Allow-Origin', '*')
                return response
        return jsonify({"message":"invalid request"}),401
    except:
        traceback.print_exc()
        return jsonify({"message":"unexpected error"}),401

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-p","--port",type=int,default=8080)
    args = parser.parse_args()
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.run(debug=True,host="0.0.0.0",port=args.port)

"""


"""