import pandas as pd


df = pd.read_csv('../csv/sample.csv')
df.index
df.info()
group_df = df.groupby(['variety', 'winery'])['description'].apply(lambda x: '; '.join(x))
group_df2 = df.groupby(['winery', 'variety'])['description'].apply(lambda x: '; '.join(x))
group_df3 = df.groupby(['variety', 'province'])['description'].apply(lambda x: '; '.join(x))

g = df.groupby(['variety', 'winery'])
g.indices
g['description'].apply(lambda x: '; '.join(x))
g.head()


group_df3

df[df['country'] == 'France']

df[df['Unnamed: 0'] == 63271]

######################################


if __name__ == '__main__':

    filename = '../csv/sample.csv'
