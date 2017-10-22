import os, json
import pandas as pd

path_to_json = 'data/'
json_files = [pos_json for pos_json in os.listdir(path_to_json) if pos_json.endswith('.json')]
print (json_files)

len(json_files)

columns = ['country', 'description', 'designation', 'points', 'price', 'province', 'region_1', 'region_2', 'url', 'variety', 'winery']
wine_df = pd.DataFrame(columns=columns)

# df = pd.read_json('data/winemag-data_1507444130.48.json')
# df.head()
# wine_df = wine_df.append(df, ignore_index=True)

for index, js in enumerate(json_files):
    with open(os.path.join(path_to_json, js)) as json_file:
        #step_json = json_file.readlines()
        print js
        df = pd.read_json(json_file)
        wine_df = wine_df.append(df, ignore_index=True)

wine_df.head()
wine_df.info()

file_name = 'winemag_data_{}.csv'.format('20171013')
wine_df.to_csv(file_name, sep=',', encoding='utf-8')
