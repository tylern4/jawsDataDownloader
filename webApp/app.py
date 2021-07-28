from flask import Flask
from flask import render_template, request
from services.getDataFromCromwell import update
import logging
from datetime import datetime
from time import sleep
import random

app = Flask(__name__)


@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"


@app.route("/update", methods=['POST', 'GET'])
def _update():
    if request.method == "POST":
        start = datetime.now()
        message = update()
        end = (datetime.now()-start).microseconds
    else:
        start = 0
        end = 0
        message = ""

    return render_template("main.html",
                           start=start,
                           end=end,
                           message=message)
