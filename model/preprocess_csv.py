'''
Used this to merge kaggle data with additional scraped data
from same winemag.com source. This merge helps preserve the
utf8 enoding.

Creates all_wine_data with ~128k unique wine reviews
'''

import pandas as pd

original_data = '../../data/winemag-data_first150k.csv'
scrape_data = '../../data/scraped_38k.csv'

og_df = pd.read_csv(original_data)
s_df = pd.read_csv(scrape_data, header=0)
s_df = s_df.drop('url', axis=1)
delthese = list(s_df[s_df['points'] == 'points'].index)
s_df.drop(delthese, axis=0, inplace=True)
s_df['points'] = s_df['points'].astype(int)
s_df['price'] = s_df['price'].astype(float)

new_df = og_df.append(s_df, ignore_index=True)
new_df = new_df.drop('Unnamed: 0', axis=1)
new_df = new_df.drop_duplicates()

new_df.to_csv('../../data/all_wine_data.csv')
print "completed"
