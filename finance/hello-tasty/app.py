import os
import sys
import json
import requests
import traceback
import argparse
from flask import (
    Flask, flash, render_template, request, redirect, url_for,
    send_file, jsonify,
)

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(THIS_DIR,'config.yml')
with open(config_path, 'r') as f:
    mycfg = yaml.safe_load(f.read())

app = Flask(__name__,
    static_url_path='', 
    static_folder='static',
    template_folder='templates',
)

@app.route('/ping')
def ping():
    logger.debug("ping")
    return jsonify({'message':'pong'})

@app.route('/')
def home():
    return render_template('home.html')

# curl -X POST https://api.cert.tastyworks.com/sessions \
# -H "X-Tastyworks-OTP: 123456" \
# -H "Content-Type: application/json" 
# -d '{ "login": "myusername", "password": "mypassword" }'
#url = 'https://www.w3schools.com/python/demopage.php'
#myobj = {'somekey': 'somevalue'}
#x = requests.post(url, json = myobj)

@app.route('/login')
def login():
    myurl = "https://api.cert.tastyworks.com/sessions"
    blob = { "login": os.environ.get("USERNAME"), "password": os.environ.get("PASSWORD") }
    resp = requests.post(myurl,json=blob)
    return render_template('home.html',status=resp.text)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-p','--port',type=int,default=5000)
    parser.add_argument('-d','--debug',action='store_true')
    args = parser.parse_args()
    app.run(host="0.0.0.0", port=args.port, debug=args.debug)
