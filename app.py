from flask import Flask, render_template, request, jsonify
import pickle
import json
import random
from build_model import RecCosineSimilarity

app = Flask(__name__)

with open('static/wine_df.pkl', 'rb') as f:
    wine_df = pickle.load(f)

with open('static/cos_sim_model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/', methods=['GET'])
def index():
    # Generate random list from wine library
    rand_idxs = random.sample(range(len(wine_df)), 10)
    wine_names = [wine_df['product'][i].decode('utf8') for i in rand_idxs]
    z_wine_names = zip(rand_idxs, wine_names)
    return render_template('index.html', wines=z_wine_names)

@app.route('/predict', methods=['POST'])
def predict():
    user_input = list(request.json['user_input'])
    data = str(user_input[0])
    rec_type = user_input[1]
    if rec_type=='1':
        rec_ids = model.recommend_to_one(int(data))
    elif rec_type=='2':
        rec_ids = model.recommend_user_input(data)
    wineList = []
    for r in rec_ids:
        wDict = {
            'id': r,
            'name': wine_df['product'][r],
            'province': wine_df['province'][r],
            'price': wine_df['price'][r]}
        wineList.append(wDict)
    recs = json.dumps(wineList, encoding='utf8')
    return jsonify({'recommendations': recs})

if __name__ == '__main__':
    app.run(host='0.0.0.0') #, debug=True)
