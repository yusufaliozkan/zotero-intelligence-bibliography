import pandas as pd

df_authors = pd.read_csv('all_items.csv')
# df_authors['FirstName2'].fillna('', inplace=True)
df_authors['Author_name'] = df_authors['FirstName2'].apply(lambda x: x.split(', ') if isinstance(x, str) and x else x)
df_authors = df_authors.explode('Author_name')
df_authors.reset_index(drop=True, inplace=True)
df_authors = df_authors.dropna(subset=['FirstName2'])
name_replacements = {
    'David Gioe': 'David V. Gioe',
    'David Vincent Gioe': 'David V. Gioe',
    'Michael Goodman': 'Michael S. Goodman',
    'Michael S Goodman': 'Michael S. Goodman',
    'Michael Simon Goodman': 'Michael S. Goodman',
    'Thomas Maguire':'Thomas J. Maguire',
    'Thomas Joseph Maguire':'Thomas J. Maguire',
    'Huw John Davies':'Huw J. Davies',
    'Huw Davies':'Huw J. Davies',
    'Philip H.J. Davies':'Philip H. J. Davies',
    'Philip Davies':'Philip H. J. Davies',
    'Dan Lomas':'Daniel W. B. Lomas',
    'Richard Aldrich':'Richard J. Aldrich',
    'Richard J Aldrich':'Richard J. Aldrich',
    'Steven Wagner':'Steven B. Wagner',
    'Daniel Larsen':'Daniel R. Larsen',
    'Daniel Richard Larsen':'Daniel R. Larsen',
    'Loch Johnson':'Loch K. Johnson',
    'Sir David Omand Gcb':'David Omand',
    'Sir David Omand':'David Omand',
    'John Ferris':'John R. Ferris',
    'John Robert Ferris':'John R. Ferris',
    'Richard Betts':'Richard K. Betts',
    'Wesley Wark':'Wesley K. Wark'
}
df_authors['Author_name'] = df_authors['Author_name'].map(name_replacements).fillna(df_authors['Author_name'])