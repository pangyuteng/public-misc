import argparse
import traceback
import os
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-p","--port",type=int,default=8080)
    args = parser.parse_args()
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.run(debug=True,host="0.0.0.0",port=args.port)

"""


"""