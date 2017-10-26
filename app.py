from flask import Flask
from flask import render_template, request, jsonify
import pickle

from rec_CosSim import RecCosineSimilarity

app = Flask(__name__)

with open('/static/cos_sim_model.pkl') as f:
    model = pickle.load(f)

with open('/static/wine_df.pkl') as f:
    wine_df = pickle.load(f)

@app.route('/', methods=['GET'])
def index():
    pass

@app.route('/recommender', methods=['GET', 'POST'])
def recommender():
    pass

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
