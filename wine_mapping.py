import pandas as pd
from cos_sim import *
import csv

'''
Working Progress
'''

filename = '../csv/sample.csv'
wine = load_data(filename)
wine = create_product_name(wine)

style = pd.read_csv('../csv/variety_style_map.csv')
style_dict = style.set_index('Variety')['Style_ID'].to_dict()

style_dict
wine.shape
wine['style'] = wine['variety']
wine.head()
wine = wine.replace({'style': style_dict})
wine.head()
wine['style'].unique()

wine.head()
for i in xrange(len(wine['variety'])):
    temp_num = style_dict.get(wine['variety'][i], 10)
    wine['style'][i] = temp_num

wine['variety'][1]
style_dict.get(wine['variety'][1], 10)

temp_cat = iferror(style_dict['Petite Sirah'], 10)
