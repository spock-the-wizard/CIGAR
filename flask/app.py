#import os
#from app import app
#import urllib.request
from flask import Flask, flash, request, redirect, url_for, render_template
#from werkzeug.utils import secure_filename

app = Flask(__name__)

@app.route('/')
def home():
   return render_template('home.html')

@app.route('/')
def get_username():
   name = request.form['UserName']
   return name

@app.route('/')
def get_userimg():
   userimg = request.form['UserImage']
   return userimg

@app.route('/keywords')
def ir_show_keywords():
   return render_template('ir_show_keywords.html')

if __name__ == "__main__":
    app.run(debug=True)