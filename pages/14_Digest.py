from pyzotero import zotero
import pandas as pd
import streamlit as st
from IPython.display import HTML
import streamlit.components.v1 as components
import numpy as np
import altair as alt
from pandas.io.json import json_normalize
import datetime
import plotly.express as px
import numpy as np
import re
import matplotlib.pyplot as plt
import nltk
nltk.download('all')
from nltk.corpus import stopwords
nltk.download('stopwords')
from wordcloud import WordCloud
from gsheetsdb import connect
import datetime as dt     


st.set_page_config(layout = "centered", 
                    page_title='Intelligence bibliography',
                    page_icon="https://images.pexels.com/photos/315918/pexels-photo-315918.png",
                    initial_sidebar_state="auto") 

st.title("Intelligence bibliography digest")

image = 'https://images.pexels.com/photos/315918/pexels-photo-315918.png'

with st.sidebar:

    st.image(image, width=150)
    st.sidebar.markdown("# Intelligence bibliography")
    with st.expander('About'):
        st.write('''This website lists secondary sources on intelligence studies and intelligence history.
        The sources are originally listed in the [Intelligence bibliography Zotero library](https://www.zotero.org/groups/2514686/intelligence_bibliography).
        This website uses [Zotero API](https://github.com/urschrei/pyzotero) to connect the *Intelligence bibliography Zotero group library*.
        To see more details about the sources, please visit the group library [here](https://www.zotero.org/groups/2514686/intelligence_bibliography/library). 
        If you need more information about Zotero, visit [this page](https://www.intelligencenetwork.org/zotero).
        ''')
        components.html(
        """
        <a rel="license" href="http://creativecommons.org/licenses/by/4.0/"><img alt="Creative Commons Licence" style="border-width:0" 
        src="https://i.creativecommons.org/l/by/4.0/80x15.png" /></a><br />
        © 2022 All rights reserved. This website is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">Creative Commons Attribution 4.0 International License</a>.
        """
        )
    with st.expander('Source code'):
        st.info('''
        Source code of this app is available [here](https://github.com/YusufAliOzkan/zotero-intelligence-bibliography).
        ''')
    with st.expander('Disclaimer'):
        st.warning('''
        This website and the Intelligence bibliography Zotero group library do not list all the sources on intelligence studies. 
        The list is created based on the creator's subjective views.
        ''')
    with st.expander('Contact us'):
        st.write('If you have any questions or suggestions, please do get in touch with us by filling the form [here](https://www.intelligencenetwork.org/contact-us).')


df_csv = pd.read_csv(r'all_items.csv', index_col=None)
df_csv['Date published'] = pd.to_datetime(df_csv['Date published'],utc=True, errors='coerce').dt.date
df_csv['Publisher'] =df_csv['Publisher'].fillna('')
df_csv['Journal'] =df_csv['Journal'].fillna('')
df_csv['firstName'] =df_csv['firstName'].fillna('')
df_csv['lastName'] =df_csv['lastName'].fillna('')

df_csv = df_csv.drop(['Unnamed: 0'], axis=1)

today = dt.date.today()
today2 = dt.date.today().strftime('%d/%m/%Y')
st.write('Intelligence bibliogrpahy digest - Day: '+ str(today2))

st.markdown('#### Contents')
st.write('[Publications](#publications)')

with st.expander('Publications:', expanded=True):
    st.header('Publications')
    container = st.container()
    previous_10 = today - dt.timedelta(days=10)
    previous_20 = today - dt.timedelta(days=20)
    previous_30 = today - dt.timedelta(days=30)
    rg = previous_30
    a=30

    range_day = st.radio('How many days do you want to go back?', ('30', '20', '10'))

    if range_day == '10':
        rg = previous_10
        a = 10
    if range_day == '20':
        rg = previous_20
        a =20
    if range_day == '30':
        rg = previous_30
        a=30

    filter = (df_csv['Date published']>rg) & (df_csv['Date published']<today)
    df_csv = df_csv.loc[filter]

    df_csv['Date published'] = pd.to_datetime(df_csv['Date published'],utc=True, errors='coerce').dt.tz_convert('Europe/London')
    df_csv['Date published new'] = df_csv['Date published'].dt.strftime('%d/%m/%Y')
    df_csv['Date published'] = df_csv['Date published'].fillna('No date')
    df_csv.sort_values(by='Date published', ascending = False, inplace=True)


    container.subheader('Sources published in the last ' + str(a) + ' days')

    sort_by_type = st.checkbox('Sort by publication type', key='type')
    types = st.multiselect('Publication type', df_csv['Publication type'].unique(),df_csv['Publication type'].unique())
    df_csv = df_csv[df_csv['Publication type'].isin(types)]

    if df_csv['Title'].any() in ("", [], None, 0, False):
        st.write('There is no publication in the last '+ str(a) +' days!')

    if sort_by_type:
        df_csv = df_csv.sort_values(by=['Publication type'], ascending = True)

        types2 = df_csv['Publication type'].unique()
        types2 = pd.DataFrame(types2, columns=['Publication type'])
        row_nu_types2 = len(types2.index)
        for i in range(row_nu_types2):
            st.subheader(types2['Publication type'].iloc[i])
            b = types2['Publication type'].iloc[i]
            df_csva = df_csv[df_csv['Publication type']==b]
            df_lasta = ('**'+ df_csva['Publication type']+ '**'+ ': ' + 
                    df_csva['Title'] + ', [Publication link]'+ '('+ df_csva['Link to publication'] + ')' +
                    ' (First author: ' + '*' + df_csva['firstName'] + '*'+ ' ' + '*' + df_csva['lastName'] + '*' + ') ' +
                    ' (Published on: ' + df_csva['Date published new'] + ')'
                    )
            row_nu = len(df_csva.index)
            for i in range(row_nu):
                st.write(''+str(i+1)+') ' +df_lasta.iloc[i])

    else:
        df_last = ('**'+ df_csv['Publication type']+ '**'+ ': ' + 
                    df_csv['Title'] + ', [Publication link]'+ '('+ df_csv['Link to publication'] + ')' +
                    ' (First author: ' + '*' + df_csv['firstName'] + '*'+ ' ' + '*' + df_csv['lastName'] + '*' + ') ' +
                    ' (Published on: ' + df_csv['Date published new'] + ')'
                    )
        row_nu = len(df_csv.index)
        for i in range(row_nu):
            st.write(''+str(i+1)+') ' +df_last.iloc[i])


with st.expander('Events', expanded=True):
    # Create a connection object.
    conn = connect()

    # Perform SQL query on the Google Sheet.
    # Uses st.cache to only rerun when the query changes or after 10 min.
    @st.cache(ttl=10)
    def run_query(query):
        rows = conn.execute(query, headers=1)
        rows = rows.fetchall()
        return rows

    sheet_url = st.secrets["public_gsheets_url"]
    rows = run_query(f'SELECT * FROM "{sheet_url}"')

    data = []
    columns = ['event_name', 'organiser', 'link', 'date', 'venue', 'details']

    # Print results.
    for row in rows:
        data.append((row.event_name, row.organiser, row.link, row.date, row.venue, row.details))

    pd.set_option('display.max_colwidth', None)
    df_gs = pd.DataFrame(data, columns=columns)
    df_gs['date_new'] = pd.to_datetime(df_gs['date'], dayfirst = True).dt.strftime('%d/%m/%Y')
    df_gs.sort_values(by='date', ascending = True, inplace=True)
    df_gs = df_gs.drop_duplicates(subset=['event_name', 'link'], keep='first')
    today = dt.date.today()
    filter = (df_gs['date']>=today)
    df_gs = df_gs.loc[filter]
    df_gs = df_gs.head(3)
    if df_gs['event_name'].any() in ("", [], None, 0, False):
        st.write('No upcoming event!')
    df_gs1 = ('['+ df_gs['event_name'] + ']'+ '('+ df_gs['link'] + ')'', organised by ' + '**' + df_gs['organiser'] + '**' + '. Date: ' + df_gs['date_new'] + ', Venue: ' + df_gs['venue'])
    row_nu = len(df_gs.index)
    for i in range(row_nu):
        st.write(''+str(i+1)+') '+ df_gs1.iloc[i])
    st.write('Visit the [Events on intelligence](https://intelligence.streamlit.app/Events) page to see more!')

st.write('---')
components.html(
"""
<a rel="license" href="http://creativecommons.org/licenses/by/4.0/"><img alt="Creative Commons Licence" style="border-width:0" 
src="https://i.creativecommons.org/l/by/4.0/80x15.png" /></a><br />
© 2022 All rights reserved. This website is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">Creative Commons Attribution 4.0 International License</a>.
"""
)
