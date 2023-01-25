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
from fpdf import FPDF
import base64


st.set_page_config(layout = "centered", 
                    page_title='Intelligence bibliography',
                    page_icon="https://images.pexels.com/photos/315918/pexels-photo-315918.png",
                    initial_sidebar_state="auto") 

st.title("Intelligence studies network digest")

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
st.write('Intelligence studies network digest - Day: '+ str(today2))

st.markdown('#### Contents')
st.caption('[Publications](#publications)')
st.caption('[Events](#events)')
st.caption('[Conferences](#conferences)')
st.caption('[Call for papers](#call-for-papers)')


with st.expander('Publications:', expanded=True):
    st.header('Publications')
    previous_10 = today - dt.timedelta(days=10)
    previous_20 = today - dt.timedelta(days=20)
    previous_30 = today - dt.timedelta(days=30)
    rg = previous_10
    a='10 days'

    range_day = st.radio('Show sources published in the last:', ('10 days', '20 days', '30 days'))

    if range_day == '10 days':
        rg = previous_10
        a = '10 days'
    if range_day == '20 days':
        rg = previous_20
        a ='20 days'
    if range_day == '30 days':
        rg = previous_30
        a='30 days'

    filter = (df_csv['Date published']>rg) & (df_csv['Date published']<=today)
    df_csv = df_csv.loc[filter]

    df_csv['Date published'] = pd.to_datetime(df_csv['Date published'],utc=True, errors='coerce').dt.tz_convert('Europe/London')
    df_csv['Date published new'] = df_csv['Date published'].dt.strftime('%d/%m/%Y')
    df_csv['Date published'] = df_csv['Date published'].fillna('No date')
    df_csv.sort_values(by='Date published', ascending = False, inplace=True)    

    sort_by_type = st.checkbox('Sort by publication type', key='type')
    types = st.multiselect('Publication type', df_csv['Publication type'].unique(),df_csv['Publication type'].unique())
    df_csv = df_csv[df_csv['Publication type'].isin(types)]

    st.subheader('Sources published in the last ' + str(a))

    if df_csv['Title'].any() in ("", [], None, 0, False):
        st.write('There is no publication published in the last '+ str(a))

    if sort_by_type:
        df_csv = df_csv.sort_values(by=['Publication type'], ascending = True)
        types2 = df_csv['Publication type'].unique()
        types2 = pd.DataFrame(types2, columns=['Publication type'])
        row_nu_types2 = len(types2.index)
        for i in range(row_nu_types2):
            st.subheader(types2['Publication type'].iloc[i])
            b = types2['Publication type'].iloc[i]
            df_csva = df_csv[df_csv['Publication type']==b]
            df_lasta = ('**'+ df_csva['Publication type']+ '**'+ ": '" + 
                    df_csva['Title'] + "'," +
                    ' (First author: ' + '*' + df_csva['firstName'] + '*'+ ' ' + '*' + df_csva['lastName'] + '*' + ') ' +
                    ' (Published in: ' + '*' + df_csva['Journal'] + '*' +')' +
                    ' (Published on: ' + df_csva['Date published new'] + ')' +
                    ", [Publication link]"+ '('+ df_csva['Link to publication'] + ')'
                    )
            row_nu = len(df_csva.index)
            for i in range(row_nu):
                st.write(''+str(i+1)+') ' +df_lasta.iloc[i])

    else:
        df_last = ('**'+ df_csv['Publication type']+ '**'+ ": '"  + 
                    df_csv['Title'] +  "',"  +
                    ' (First author: ' + '*' + df_csv['firstName'] + '*'+ ' ' + '*' + df_csv['lastName'] + '*' + ') ' +
                    ' (Published in: ' + '*' + df_csv['Journal'] +'*' + ')' +
                    ' (Published on: ' + df_csv['Date published new'] + ')'+
                    ", [Publication link]"+ '('+ df_csv['Link to publication'] + ')'
                    )
        row_nu = len(df_csv.index)
        for i in range(row_nu):
            st.write(''+str(i+1)+') ' +df_last.iloc[i])

st.caption('[Go to top](#intelligence-studies-network-digest)')

with st.expander('Events:', expanded=True):
    st.header('Events')
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

    next_10 = today + dt.timedelta(days=10)    
    next_20 = today + dt.timedelta(days=20)
    next_30 = today + dt.timedelta(days=30)
    rg2 = next_10
    aa='10 days'
    range_day = st.radio('Show events in the next:', ('10 days', '20 days', '30 days'), key='events')
    if range_day == '10 days':
        rg2 = next_10
        aa = '10 days'
    if range_day == '20 days':
        rg2 = next_20
        aa ='20 days'
    if range_day == '30 days':
        rg2 = next_30
        aa='30 days'
    filter_events = (df_gs['date']<rg2) & (df_gs['date']>=today)
    df_gs = df_gs.loc[filter_events]

    st.subheader('Events in the next ' + str(aa))
    display = st.checkbox('Show details')

    if df_gs['event_name'].any() in ("", [], None, 0, False):
        st.write('No upcoming event!')
    df_gs1 = ('['+ df_gs['event_name'] + ']'+ '('+ df_gs['link'] + ')'', organised by ' + '**' + df_gs['organiser'] + '**' + '. Date: ' + df_gs['date_new'] + ', Venue: ' + df_gs['venue'])
    row_nu = len(df_gs.index)
    for i in range(row_nu):
        st.write(''+str(i+1)+') '+ df_gs1.iloc[i])
        if display:
            st.caption('Details:'+'\n '+ df_gs['details'].iloc[i])
    st.write('Visit the [Events on intelligence](https://intelligence.streamlit.app/Events) page to see more!')

st.caption('[Go to top](#intelligence-studies-network-digest)')

with st.expander('Conferences:', expanded=True):
    st.header('Conferences')
    sheet_url2 = st.secrets["public_gsheets_url2"]
    rows = run_query(f'SELECT * FROM "{sheet_url2}"')

    data = []
    columns = ['conference_name', 'organiser', 'link', 'date', 'date_end', 'venue', 'details', 'location']

    # Print results.
    for row in rows:
        data.append((row.conference_name, row.organiser, row.link, row.date, row.date_end, row.venue, row.details, row.location))

    pd.set_option('display.max_colwidth', None)
    df_con = pd.DataFrame(data, columns=columns)

    df_con['date_new'] = pd.to_datetime(df_con['date'], dayfirst = True).dt.strftime('%d/%m/%Y')
    df_con['date_new_end'] = pd.to_datetime(df_con['date_end'], dayfirst = True).dt.strftime('%d/%m/%Y')
    df_con.sort_values(by='date', ascending = True, inplace=True)

    next_1mo = today + dt.timedelta(days=30)
    next_3mo = today + dt.timedelta(days=90)    
    next_6mo = today + dt.timedelta(days=180)
    rg3 = next_3mo

    range_day = st.radio('Show conferences in the next: ', ('1 month', '3 months', '6 months'), key='conferences')
    if range_day == '1 month':
        rg3 = next_1mo
        aaa = '1 month'
    if range_day == '3 months':
        rg3 = next_3mo
        aaa = '3 months'
    if range_day == '6 months':
        rg3 = next_6mo
        aaa = '6 months'
    filter_events = (df_con['date']<rg3) & (df_con['date']>=today)
    df_con = df_con.loc[filter_events]

    df_con['details'] = df_con['details'].fillna('No details')
    df_con['location'] = df_con['location'].fillna('No details')

    st.subheader('Conferences in the next ' + str(aaa))
    display = st.checkbox('Show details', key='conference')

    df_con1 = ('['+ df_con['conference_name'] + ']'+ '('+ df_con['link'] + ')'', organised by ' + '**' + df_con['organiser'] + '**' + '. Date(s): ' + df_con['date_new'] + ' - ' + df_con['date_new_end'] + ', Venue: ' + df_con['venue'])
    row_nu = len(df_con.index)
    for i in range(row_nu):
        st.write(''+str(i+1)+') '+ df_con1.iloc[i])
        if display:
            st.caption('Conference place:'+'\n '+ df_con['location'].iloc[i])
            st.caption('Details:'+'\n '+ df_con['details'].iloc[i])

st.caption('[Go to top](#intelligence-studies-network-digest)')

with st.expander('Call for papers:', expanded=True):
    st.header('Call for papers')
    sheet_url3 = st.secrets["public_gsheets_url3"]
    rows = run_query(f'SELECT * FROM "{sheet_url3}"')

    data = []
    columns = ['name', 'organiser', 'link', 'date', 'details']

    # Print results.
    for row in rows:
        data.append((row.name, row.organiser, row.link, row.deadline, row.details))

    pd.set_option('display.max_colwidth', None)
    df_cfp = pd.DataFrame(data, columns=columns)

    df_cfp['date_new'] = pd.to_datetime(df_cfp['date'], dayfirst = True).dt.strftime('%d/%m/%Y')
    df_cfp.sort_values(by='date', ascending = True, inplace=True)

    df_cfp['details'] = df_cfp['details'].fillna('No details')
    
    display = st.checkbox('Show details', key='cfp')

    filter = (df_cfp['date']>=today)
    df_cfp = df_cfp.loc[filter]
    if df_cfp['name'].any() in ("", [], None, 0, False):
        st.write('No upcoming Call for papers!')

    df_cfp1 = ('['+ df_cfp['name'] + ']'+ '('+ df_cfp['link'] + ')'', organised by '  + df_cfp['organiser'] + '. ' +'**' + 'Deadline: ' + df_cfp['date_new']+'**' )
    row_nu = len(df_cfp.index)
    for i in range(row_nu):
        st.write(''+str(i+1)+') '+ df_cfp1.iloc[i])
        if display:
            st.caption('Details:'+'\n '+ df_cfp['details'].iloc[i])

st.caption('[Go to top](#intelligence-studies-network-digest)')

st.write('---')

components.html(
"""
<a rel="license" href="http://creativecommons.org/licenses/by/4.0/"><img alt="Creative Commons Licence" style="border-width:0" 
src="https://i.creativecommons.org/l/by/4.0/80x15.png" /></a><br />
© 2022 All rights reserved. This website is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">Creative Commons Attribution 4.0 International License</a>.
"""
)
