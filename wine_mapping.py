import pandas as pd
from cos_sim import *
import csv

'''
Working Progress
'''

filename = '../csv/sample.csv'
wine = pd.read_csv(filename, encoding='latin1')
wine = create_product_name(wine)

style = pd.read_csv('../csv/variety_style_map.csv', encoding='latin1')
style_dict = style.set_index('Variety')['Style_ID'].to_dict()

style_dict
wine.shape
wine['style'] = wine['variety']
wine.head()

wine = wine.replace({'style': style_dict})
wine.head()
wine['style'].unique()
