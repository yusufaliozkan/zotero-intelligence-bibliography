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


st.set_page_config(layout = "wide", 
                    page_title='Intelligence bibliography',
                    page_icon="https://images.pexels.com/photos/315918/pexels-photo-315918.png",
                    initial_sidebar_state="auto") 

st.title("Events about intelligence")
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


today = dt.date.today()
today2 = dt.date.today().strftime('%d/%m/%Y')
st.write('Today is: '+ str(today2))

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
columns = ['event_name', 'organiser', 'link', 'date', 'venue']

# Print results.
for row in rows:
    data.append((row.event_name, row.organiser, row.link, row.date, row.venue))

pd.set_option('display.max_colwidth', None)
df_gs = pd.DataFrame(data, columns=columns)

df_gs['date_new'] = pd.to_datetime(df_gs['date'], dayfirst = True).dt.strftime('%d/%m/%Y')
df_gs['month'] = pd.to_datetime(df_gs['date'], dayfirst = True).dt.strftime('%m')
df_gs.sort_values(by='date', ascending = True, inplace=True)
# df_gs['month'] = df_gs['month'].replace('01', 'January')
# df_gs['month'] = df_gs['month'].replace('02', 'February')
# df_gs['month'] = df_gs['month'].replace('03', 'March')
# df_gs['month'] = df_gs['month'].replace('04', 'April')
# df_gs['month'] = df_gs['month'].replace('05', 'May')
# df_gs['month'] = df_gs['month'].replace('06', 'June')
# df_gs['month'] = df_gs['month'].replace('07', 'July')
# df_gs['month'] = df_gs['month'].replace('08', 'August')
# df_gs['month'] = df_gs['month'].replace('09', 'September')
# df_gs['month'] = df_gs['month'].replace('10', 'October')
# df_gs['month'] = df_gs['month'].replace('11', 'November')
# df_gs['month'] = df_gs['month'].replace('12', 'December')
online_event = st.checkbox('Show online events only')

if online_event:
    df_gs = df_gs[df_gs['venue']=='Online event']

filter = (df_gs['date']>=today)
filter2 = (df_gs['date']<today)
df_gs2 = df_gs.loc[filter2]
df_gs = df_gs.loc[filter]
if df_gs['event_name'].any() in ("", [], None, 0, False):
    st.write('No upcoming event!')

for i in range(12):
    a=i+1
    a

if '01' in df_gs['month'].values:
    st.markdown('#### Events in January')
    jan = df_gs[df_gs['month']=='01']
    df_gs1 = ('['+ jan['event_name'] + ']'+ '('+ jan['link'] + ')'', organised by ' + '**' + jan['organiser'] + '**' + '. Date: ' + jan['date_new'] + ', Venue: ' + jan['venue'])
    row_nu = len(jan.index)
    for i in range(row_nu):
        st.write(''+str(i+1)+') '+ df_gs1.iloc[i])

if '02' in df_gs['month'].values:
    st.markdown('#### Events in February')
    feb = df_gs[df_gs['month']=='02']
    df_gs1 = ('['+ feb['event_name'] + ']'+ '('+ feb['link'] + ')'', organised by ' + '**' + feb['organiser'] + '**' + '. Date: ' + feb['date_new'] + ', Venue: ' + feb['venue'])
    row_nu = len(feb.index)
    for i in range(row_nu):
        st.write(''+str(i+1)+') '+ df_gs1.iloc[i]) 

if '03' in df_gs['month'].values:
    st.markdown('#### Events in March', expanded=True)
    mar = df_gs[df_gs['month']=='03']
    df_gs1 = ('['+ mar['event_name'] + ']'+ '('+ mar['link'] + ')'', organised by ' + '**' + mar['organiser'] + '**' + '. Date: ' + mar['date_new'] + ', Venue: ' + mar['venue'])
    row_nu = len(mar.index)
    for i in range(row_nu):
        st.write(''+str(i+1)+') '+ df_gs1.iloc[i]) 

st.header('Past events')
with st.expander('Expand to see the list'):
    row_nu2 = len(df_gs2.index)
    df_gs3 = ('['+ df_gs2['event_name'] + ']'+ '('+ df_gs2['link'] + ')'', organised by ' + '**' + df_gs2['organiser'] + '**' + '. Date: ' + df_gs2['date_new'] + ', Venue: ' + df_gs2['venue'])
    row_nu = len(df_gs.index)
    for i in range(row_nu2):
        st.write(''+str(i+1)+') '+ df_gs3.iloc[i]) 


components.html(
"""
<a rel="license" href="http://creativecommons.org/licenses/by/4.0/"><img alt="Creative Commons Licence" style="border-width:0" 
src="https://i.creativecommons.org/l/by/4.0/80x15.png" /></a><br />
© 2022 All rights reserved. This website is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">Creative Commons Attribution 4.0 International License</a>.
"""
)
