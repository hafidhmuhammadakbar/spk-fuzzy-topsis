from flask import Flask, render_template
from src import fuzzy_topsis as ft

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/fuzzy', methods=['GET'])
def fuzzy():
    return render_template('fuzzy.html')

if __name__ == '__main__':
    app.run(debug=True)