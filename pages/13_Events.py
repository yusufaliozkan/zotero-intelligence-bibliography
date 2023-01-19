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

st.title("Events on intelligence")
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

tab1, tab2, tab3 = st.tabs(['Events', 'Conferences','Call for papers'])
with tab1:
    st.subheader('Events')
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
    df_gs['month'] = pd.to_datetime(df_gs['date'], dayfirst = True).dt.strftime('%m')
    df_gs['month_year'] = pd.to_datetime(df_gs['date'], dayfirst = True).dt.strftime('%m-%Y')
    df_gs.sort_values(by='date', ascending = True, inplace=True)
    df_gs = df_gs.drop_duplicates(subset=['event_name', 'link'], keep='first')
    
    df_gs['details'] = df_gs['details'].fillna('No details')
    
    online_event = st.checkbox('Show online events only')

    if online_event:
        df_gs = df_gs[df_gs['venue']=='Online event']

    display = st.checkbox('Show details')

    filter = (df_gs['date']>=today)
    filter2 = (df_gs['date']<today)
    df_gs2 = df_gs.loc[filter2]
    df_gs = df_gs.loc[filter]
    if df_gs['event_name'].any() in ("", [], None, 0, False):
        st.write('No upcoming event!')

    if '01' in df_gs['month'].values:
        st.markdown('#### Events in January')
        mon = df_gs[df_gs['month']=='01']
        df_gs1 = ('['+ mon['event_name'] + ']'+ '('+ mon['link'] + ')'', organised by ' + '**' + mon['organiser'] + '**' + '. Date: ' + mon['date_new'] + ', Venue: ' + mon['venue'])
        row_nu = len(mon.index)
        for i in range(row_nu):
            st.write(''+str(i+1)+') '+ df_gs1.iloc[i])
            if display:
                st.caption('Details:'+'\n '+ df_gs['details'].iloc[i])

    if '02' in df_gs['month'].values:
        st.markdown('#### Events in February')
        mon = df_gs[df_gs['month']=='02']
        df_gs1 = ('['+ mon['event_name'] + ']'+ '('+ mon['link'] + ')'', organised by ' + '**' + mon['organiser'] + '**' + '. Date: ' + mon['date_new'] + ', Venue: ' + mon['venue'])
        row_nu = len(mon.index)
        for i in range(row_nu):
            st.write(''+str(i+1)+') '+ df_gs1.iloc[i])
            if display:
                st.caption('Details:'+'\n '+ df_gs['details'].iloc[i])

    if '03' in df_gs['month'].values:
        st.markdown('#### Events in March')
        mon = df_gs[df_gs['month']=='03']
        df_gs1 = ('['+ mon['event_name'] + ']'+ '('+ mon['link'] + ')'', organised by ' + '**' + mon['organiser'] + '**' + '. Date: ' + mon['date_new'] + ', Venue: ' + mon['venue'])
        row_nu = len(mon.index)
        for i in range(row_nu):
            st.write(''+str(i+1)+') '+ df_gs1.iloc[i])
            if display:
                st.caption('Details:'+'\n '+ df_gs['details'].iloc[i])

    if '04' in df_gs['month'].values:
        st.markdown('#### Events in April')
        mon = df_gs[df_gs['month']=='04']
        df_gs1 = ('['+ mon['event_name'] + ']'+ '('+ mon['link'] + ')'', organised by ' + '**' + mon['organiser'] + '**' + '. Date: ' + mon['date_new'] + ', Venue: ' + mon['venue'])
        row_nu = len(mon.index)
        for i in range(row_nu):
            st.write(''+str(i+1)+') '+ df_gs1.iloc[i])
            if display:
                st.caption('Details:'+'\n '+ df_gs['details'].iloc[i])

    if '05' in df_gs['month'].values:
        st.markdown('#### Events in May')
        mon = df_gs[df_gs['month']=='05']
        df_gs1 = ('['+ mon['event_name'] + ']'+ '('+ mon['link'] + ')'', organised by ' + '**' + mon['organiser'] + '**' + '. Date: ' + mon['date_new'] + ', Venue: ' + mon['venue'])
        row_nu = len(mon.index)
        for i in range(row_nu):
            st.write(''+str(i+1)+') '+ df_gs1.iloc[i])
            if display:
                st.caption('Details:'+'\n '+ df_gs['details'].iloc[i])

    if '06' in df_gs['month'].values:
        st.markdown('#### Events in June')
        mon = df_gs[df_gs['month']=='06']
        df_gs1 = ('['+ mon['event_name'] + ']'+ '('+ mon['link'] + ')'', organised by ' + '**' + mon['organiser'] + '**' + '. Date: ' + mon['date_new'] + ', Venue: ' + mon['venue'])
        row_nu = len(mon.index)
        for i in range(row_nu):
            st.write(''+str(i+1)+') '+ df_gs1.iloc[i])
            if display:
                st.caption('Details:'+'\n '+ df_gs['details'].iloc[i])

    if '07' in df_gs['month'].values:
        st.markdown('#### Events in July')
        mon = df_gs[df_gs['month']=='07']
        df_gs1 = ('['+ mon['event_name'] + ']'+ '('+ mon['link'] + ')'', organised by ' + '**' + mon['organiser'] + '**' + '. Date: ' + mon['date_new'] + ', Venue: ' + mon['venue'])
        row_nu = len(mon.index)
        for i in range(row_nu):
            st.write(''+str(i+1)+') '+ df_gs1.iloc[i])
            if display:
                st.caption('Details:'+'\n '+ df_gs['details'].iloc[i])

    if '08' in df_gs['month'].values:
        st.markdown('#### Events in August')
        mon = df_gs[df_gs['month']=='08']
        df_gs1 = ('['+ mon['event_name'] + ']'+ '('+ mon['link'] + ')'', organised by ' + '**' + mon['organiser'] + '**' + '. Date: ' + mon['date_new'] + ', Venue: ' + mon['venue'])
        row_nu = len(mon.index)
        for i in range(row_nu):
            st.write(''+str(i+1)+') '+ df_gs1.iloc[i])
            if display:
                st.caption('Details:'+'\n '+ df_gs['details'].iloc[i])

    if '09' in df_gs['month'].values:
        st.markdown('#### Events in September')
        mon = df_gs[df_gs['month']=='09']
        df_gs1 = ('['+ mon['event_name'] + ']'+ '('+ mon['link'] + ')'', organised by ' + '**' + mon['organiser'] + '**' + '. Date: ' + mon['date_new'] + ', Venue: ' + mon['venue'])
        row_nu = len(mon.index)
        for i in range(row_nu):
            st.write(''+str(i+1)+') '+ df_gs1.iloc[i])
            if display:
                st.caption('Details:'+'\n '+ df_gs['details'].iloc[i])

    if '10' in df_gs['month'].values:
        st.markdown('#### Events in October')
        mon = df_gs[df_gs['month']=='10']
        df_gs1 = ('['+ mon['event_name'] + ']'+ '('+ mon['link'] + ')'', organised by ' + '**' + mon['organiser'] + '**' + '. Date: ' + mon['date_new'] + ', Venue: ' + mon['venue'])
        row_nu = len(mon.index)
        for i in range(row_nu):
            st.write(''+str(i+1)+') '+ df_gs1.iloc[i])
            if display:
                st.caption('Details:'+'\n '+ df_gs['details'].iloc[i])

    if '11' in df_gs['month'].values:
        st.markdown('#### Events in November')
        mon = df_gs[df_gs['month']=='11']
        df_gs1 = ('['+ mon['event_name'] + ']'+ '('+ mon['link'] + ')'', organised by ' + '**' + mon['organiser'] + '**' + '. Date: ' + mon['date_new'] + ', Venue: ' + mon['venue'])
        row_nu = len(mon.index)
        for i in range(row_nu):
            st.write(''+str(i+1)+') '+ df_gs1.iloc[i])
            if display:
                st.caption('Details:'+'\n '+ df_gs['details'].iloc[i])

    if '12' in df_gs['month'].values:
        st.markdown('#### Events in December')
        mon = df_gs[df_gs['month']=='12']
        df_gs1 = ('['+ mon['event_name'] + ']'+ '('+ mon['link'] + ')'', organised by ' + '**' + mon['organiser'] + '**' + '. Date: ' + mon['date_new'] + ', Venue: ' + mon['venue'])
        row_nu = len(mon.index)
        for i in range(row_nu):
            st.write(''+str(i+1)+') '+ df_gs1.iloc[i])
            if display:
                st.caption('Details:'+'\n '+ df_gs['details'].iloc[i])

    st.header('Past events')
    with st.expander('Expand to see the list'):
        row_nu2 = len(df_gs2.index)
        df_gs3 = ('['+ df_gs2['event_name'] + ']'+ '('+ df_gs2['link'] + ')'', organised by ' + '**' + df_gs2['organiser'] + '**' + '. Date: ' + df_gs2['date_new'] + ', Venue: ' + df_gs2['venue'])
        row_nu = len(df_gs.index)
        for i in range(row_nu2):
            st.write(''+str(i+1)+') '+ df_gs3.iloc[i])
    
    st.header('Event visuals')
    df_gs
    date_plot=df_gs['month_year'].value_counts()
    date_plot=date_plot.reset_index()
    date_plot=date_plot.rename(columns={'index':'Date','month_year':'Count'})
    date_plot
    fig = px.bar(date_plot, x='', y='Count')
    fig.update_xaxes(tickangle=-70)
    fig.update_layout(
        autosize=False,
        width=400,
        height=500,)
    fig.update_layout(title={'text':'Events over time', 'y':0.95, 'x':0.5, 'yanchor':'top'})
    st.plotly_chart(fig, use_container_width = True)

with tab2:
    st.subheader('Conferences')
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

    df_con['details'] = df_con['details'].fillna('No details')
    df_con['location'] = df_con['location'].fillna('No details')
    
    display = st.checkbox('Show details', key='conference')

    filter = (df_con['date']>=today)
    df_con = df_con.loc[filter]
    if df_con['conference_name'].any() in ("", [], None, 0, False):
        st.write('No upcoming conference!')

    df_con1 = ('['+ df_con['conference_name'] + ']'+ '('+ df_con['link'] + ')'', organised by ' + '**' + df_con['organiser'] + '**' + '. Date(s): ' + df_con['date_new'] + ' - ' + df_con['date_new_end'] + ', Venue: ' + df_con['venue'])
    row_nu = len(df_con.index)
    for i in range(row_nu):
        st.write(''+str(i+1)+') '+ df_con1.iloc[i])
        if display:
            st.caption('Conference place:'+'\n '+ df_con['location'].iloc[i])
            st.caption('Details:'+'\n '+ df_con['details'].iloc[i])
        
with tab3:
    st.subheader('Call for papers')
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
    df_con.sort_values(by='date', ascending = True, inplace=True)

    df_cfp['details'] = df_cfp['details'].fillna('No details')
    
    display = st.checkbox('Show details', key='cfp')

    filter = (df_cfp['date']>=today)
    df_cfp = df_cfp.loc[filter]
    if df_cfp['name'].any() in ("", [], None, 0, False):
        st.write('No upcoming Call for papers!')

    df_cfp1 = ('['+ df_cfp['name'] + ']'+ '('+ df_cfp['link'] + ')'', organised by ' + '**' + df_cfp['organiser'] + '**' + '. Deadline: ' + df_cfp['date_new'])
    row_nu = len(df_cfp.index)
    for i in range(row_nu):
        st.write(''+str(i+1)+') '+ df_cfp1.iloc[i])
        if display:
            st.caption('Details:'+'\n '+ df_cfp['details'].iloc[i])

st.write('---')
components.html(
"""
<a rel="license" href="http://creativecommons.org/licenses/by/4.0/"><img alt="Creative Commons Licence" style="border-width:0" 
src="https://i.creativecommons.org/l/by/4.0/80x15.png" /></a><br />
© 2022 All rights reserved. This website is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">Creative Commons Attribution 4.0 International License</a>.
"""
)
