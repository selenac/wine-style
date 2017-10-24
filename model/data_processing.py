import pandas as pd
from nltk.corpus import stopwords

# TODO How to handle the accented words in dataset? Latin-1 errors.

def clean_data(filepath):
    '''
    Input:
        filepath: The data file's path
    Output:
        wine_df: data as pandas dataframe with additional product column
        wine_stop_library: list of Stop words library
    '''
    wine_df = pd.read_csv(filepath)
    wine_df = _create_product_col(wine_df)
    wine_stop_library = _create_wine_stop(wine_df)
    return wine_df, wine_stop_library

def agg_description(df):
    '''
    Concatenate the descriptions based on winery and variety.
    Input: Dataframe
    Output:
        agg_df: New dataframe with columns
            variety", "winery", "description" aggregated descriptions
        groups: dictionary with indices result from group
    '''
    # dictionary with index, use .tolist() to extract
    groups = df.groupby(['variety', 'winery']).groups
    agg_df = df.groupby(['variety', 'winery'])['description'].apply(lambda x: '; '.join(x))
    agg_df.reset_index()
    return agg_df, groups

#########################################################

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
                     'yet', 'seem', 'bottle', 'flavor', 'show', 'good']
    return stopwords.words('english') + wine_stop_lib + _create_variety_list(df)

if __name__ == '__main__':
    filepath = '../data/sample.csv'
    df, stop_lib = clean_data(filepath)
