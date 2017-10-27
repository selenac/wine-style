from flask import Flask, render_template, request, jsonify
import pickle
import json
from build_model import RecCosineSimilarity

app = Flask(__name__)

with open('static/cos_sim_model.pkl') as f:
    model = pickle.load(f)

with open('static/wine_df.pkl') as f:
    wine_df = pickle.load(f)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')
#
# @app.route('/bybottle', methods=['GET'])
# def bybottle():
#     pass
#
# @app.route('/bydescription', methods=['GET'])
# def bydescription():
#     pass
#

@app.route('/predict', methods=['POST'])
def predict():
    # if inquiry == 'bybottle': #TODO add this somewhere...
    #     wine_name = request.json['wine_name']
    #     wine_id = wine_df.index[wine_df['product']==wine_name][0]
    #     rec_ids = model.recommend_to_one(wine_id)
    # else:
    data = str(request.json['user_input'])
    rec_ids = model.recommend_user_input(data)
    wineList = []
    for r in rec_ids:
        wDict = {
            'Id': r,
            'Name': wine_df['product'][r],
            'Country': wine_df['country'][r],
            'Province': wine_df['province'][r],
            'Price': wine_df['price'][r]}
        wineList.append(wDict)
    recs = json.dumps(wineList)
    return jsonify({'recommendations': recs})



    # render_template('')
    #if recommend to one
        #select from wine product list + submit
        # display traits of wine selected + description
        #return user_wine wine_id for wine submitted
        # recs = cs.recommend_to_one(wine_id=user_wine)
        # for idx in top_n_recs:
        #     wine_df['product'][idx]
        #     wine_df['province'][idx]
        #     wine_df['country'][idx]
        #     wine_df['price'][idx]
        #

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
