import pandas as pd
from nltk.corpus import stopwords

# TODO How to handle the accented words in dataset? Latin-1 errors.

def clean_data(filepath, agg_desc=0):
    '''
    Input:
        filepath: The data file's path
        agg_desc: if 1 - Aggregate descriptions with grouping winery/variety
    Output:
        wine_df: data as pandas dataframe with additional product column
        wine_stop_library: list of Stop words library
        groups: if agg_desc=1, return grouped indices to track wine id
    '''
    wine_df = pd.read_csv(filepath)
    wine_df = _create_product_col(wine_df)
    wine_stop_library = _create_wine_stop(wine_df)
    if agg_desc == 1:
        groups, agg_df = agg_description(wine_df)
        return agg_df, wine_stop_library, groups
    return wine_df, wine_stop_library

def _create_product_col(df):
    '''
    Create wine product column with name
    '''
    df['product'] = df['winery'] + ' ' + df['designation'].fillna('') + ' ' + df['variety']
    return df

def _create_variety_list(df):
    '''
    Output: Unique variety list
    '''
    return map(str.lower, df['variety'].unique())

def _create_wine_stop(df):
    '''
    Combine standard english stops with wine specific text
    Output: Wine stop words to use in tokenizing
    '''
    # working list of stop words
    wine_stop_lib = ['aromas', 'drink', 'fruit', 'palate', 'wine', 'like', 'bit',
                     'flavor', 'fine', 'sense', 'note', 'notes', 'frame', 'alcohol',
                     'yet', 'seem', 'bottle', 'flavor', ]
    return stopwords.words('english') + wine_stop_lib + _create_variety_list(df)

#########################################################

def agg_description(df):
    '''
    Concatenate the descriptions based on winery and variety.
    Input: Dataframe
    Output:
        groups: dictionary with indices result from group
        agg_df: New dataframe with columns
            variety", "winery", "description" aggregated descriptions
    '''
    # dictionary with index, use .tolist() to extract
    groups = df.groupby(['variety', 'winery']).groups
    # agg_df[('Zinfandel', 'Temptation')] to access description
    agg_df = df.groupby(['variety', 'winery'])['description'].apply(lambda x: '; '.join(x))
    agg_df.reset_index()
    return groups, agg_df

if __name__ == '__main__':
    filepath = '../data/sample.csv'
    df, stop_lib = clean_data(filepath)
