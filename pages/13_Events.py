from pyzotero import zotero
import pandas as pd
import streamlit as st
from IPython.display import HTML
import streamlit.components.v1 as components
import numpy as np
import altair as alt
# from pandas.io.json import json_normalize
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
import datetime as dt
from urllib.parse import urlparse
from sidebar_content import sidebar_content
from streamlit_gsheets import GSheetsConnection

st.set_page_config(layout = "centered", 
                    page_title='Intelligence studies network',
                    page_icon="https://images.pexels.com/photos/315918/pexels-photo-315918.png",
                    initial_sidebar_state="auto") 

st.title("Intelligence studies network")
st.header("Events on intelligence")
image = 'https://images.pexels.com/photos/315918/pexels-photo-315918.png'


sidebar_content()

today = dt.date.today()
today2 = dt.date.today().strftime('%d/%m/%Y')
st.write('Today is: '+ str(today2))
container = st.container()
with st.popover('Download events data'):
    st.write('''
    The data for all events, conferences, and calls for papers is available on Zenodo. 
    Use the following link to access the dataset:

    Ozkan, Yusuf A. ‘Intelligence Studies Network Dataset’. Zenodo, 15 August 2024. https://doi.org/10.5281/zenodo.13325698.
    ''')

# Create a connection object.
conn = st.connection("gsheets", type=GSheetsConnection)
df_gs = conn.read(spreadsheet='https://docs.google.com/spreadsheets/d/10ezNUOUpzBayqIMJWuS_zsvwklxP49zlfBWsiJI6aqI/edit#gid=0')
df_gs['organiser'] = df_gs['organiser'].replace('North American Society for Intelligence History (NASIH)', 'The Society for Intelligence History (SIH)')

df_forms = conn.read(spreadsheet='https://docs.google.com/spreadsheets/d/10ezNUOUpzBayqIMJWuS_zsvwklxP49zlfBWsiJI6aqI/edit#gid=1941981997')
df_forms = df_forms.rename(columns={'Event name':'event_name', 'Event organiser':'organiser','Link to the event':'link','Date of event':'date', 'Event venue':'venue', 'Details':'details'})
df_forms['organiser'] = df_forms['organiser'].replace('North American Society for Intelligence History (NASIH)','The Society for Intelligence History (SIH)')

tab1, tab2, tab3 = st.tabs(['Events', 'Conferences','Call for papers'])
with tab1:
    st.header('Events')

    # Convert and format dates in df_gs

    df_gs['date'] = pd.to_datetime(df_gs['date'])
    df_gs['date_new'] = df_gs['date'].dt.strftime('%Y-%m-%d')
    df_gs['month'] = df_gs['date'].dt.strftime('%m')
    df_gs['year'] = df_gs['date'].dt.strftime('%Y')
    df_gs['month_year'] = df_gs['date'].dt.strftime('%Y-%m')

    # Convert and format dates in df_forms
    df_forms['date'] = pd.to_datetime(df_forms['date'])
    df_forms['date_new'] = df_forms['date'].dt.strftime('%Y-%m-%d')
    df_forms['month'] = df_forms['date'].dt.strftime('%m')
    df_forms['year'] = df_forms['date'].dt.strftime('%Y')
    df_forms['month_year'] = df_forms['date'].dt.strftime('%Y-%m')
    df_forms.sort_values(by='date', ascending=True, inplace=True)
    df_forms = df_forms.drop_duplicates(subset=['event_name', 'link', 'date'], keep='first')
    df_forms2 = df_forms.copy()
    df_forms2.sort_values(by='date_new', ascending = False, inplace=True)
    df_forms2['Timestamp'] = pd.to_datetime(df_forms2['Timestamp'], format='%m/%d/%Y %H:%M:%S')
    df_forms2['Timestamp'] = df_forms2['Timestamp'].dt.strftime('%d-%m-%Y %H:%M')
    container.write('The events page last updated on ' + '**'+ df_forms2.iloc[0]['Timestamp']+'**')
    
    df_forms['date_new'] = pd.to_datetime(df_forms['date'], dayfirst = True).dt.strftime('%d/%m/%Y')
    df_forms['month'] = pd.to_datetime(df_forms['date'], dayfirst = True).dt.strftime('%m')
    df_forms['year'] = pd.to_datetime(df_forms['date'], dayfirst = True).dt.strftime('%Y')
    df_forms['month_year'] = pd.to_datetime(df_forms['date'], dayfirst = True).dt.strftime('%Y-%m')
    df_forms.sort_values(by='date', ascending = True, inplace=True)
    df_forms = df_forms.drop_duplicates(subset=['event_name', 'link', 'date'], keep='first')
 
    
    df_forms['details'] = df_forms['details'].fillna('No details')
    df_forms = df_forms.fillna('')
    df_forms = df_forms.sort_index(ascending=True)
    df_gs = pd.concat([df_gs, df_forms], axis=0)
    df_gs = df_gs.reset_index(drop=True)
    df_gs = df_gs.drop_duplicates(subset=['event_name', 'link', 'date'], keep='first')
    df_gs_plot = df_gs.copy()
        
    col1, col2 = st.columns(2)

    with col1:
        online_event = st.checkbox('Show online events only')
        if online_event:
            online = ['Online event', 'Hybrid event']
            df_gs = df_gs[df_gs['venue'].isin(online)]
        display = st.checkbox('Show details')      
    
    with col2:
        sort_by = st.radio('Sort by', ['Date', 'Most recently added', 'Organiser'])
        
    st.write('See [📊 Event visuals](#event-visuals)')

    df_gs['date'] = pd.to_datetime(df_gs['date'], dayfirst=True)
    filter = df_gs['date']>=pd.to_datetime(today)
    filter2 = df_gs['date']<pd.to_datetime(today)
    df_gs2 = df_gs.loc[filter2]
    df_gs = df_gs.loc[filter]
    if df_gs['event_name'].any() in ("", [], None, 0, False):
        st.write('No upcoming event!')

    if sort_by == 'Most recently added':
        df_gs = df_gs.sort_index(ascending=False)
        df_last = ('['+ df_gs['event_name'] + ']'+ '('+ df_gs['link'] + ')'', organised by ' + '**' + df_gs['organiser'] + '**' + '. Date: ' + df_gs['date_new'] + ', Venue: ' + df_gs['venue'])
        row_nu = len(df_gs.index)
        for i in range(row_nu):
            st.write(''+str(i+1)+') '+ df_last.iloc[i])
            if display:
                st.caption('Details:'+'\n '+ df_gs['details'].iloc[i])

    if sort_by == 'Organiser':
        organisers = df_gs['organiser'].unique()
        organisers = pd.DataFrame(organisers, columns=['Organisers'])
        row_nu_organisers = len(organisers.index)
        for i in range(row_nu_organisers):
            st.markdown('#### '+ organisers['Organisers'].iloc[i])
            # st.subheader(organisers['Organisers'].iloc[i])
            c = organisers['Organisers'].iloc[i]
            df_o = df_gs[df_gs['organiser']==c]
            df_last = ('['+ df_o['event_name'] + ']'+ '('+ df_o['link'] + ')'', organised by ' + '**' + df_o['organiser'] + '**' + '. Date: ' + df_o['date_new'] + ', Venue: ' + df_o['venue'])
            row_nu =len(df_o.index)
            for j in range(row_nu):
                st.write(''+str(j+1)+') ' +df_last.iloc[j])
                df_last.fillna('')
                if display:
                    st.caption('Details:'+'\n '+ df_o['details'].iloc[j])

    if sort_by == 'Date':
        df_gs['date'] = pd.to_datetime(df_gs['date'])

        # Sort the DataFrame by the date
        df_gs.sort_values(by='date', ascending=True, inplace=True)

        # Create a column for the month name
        df_gs['month_name'] = df_gs['date'].dt.strftime('%B')

        # Create a column for the year
        df_gs['year'] = df_gs['date'].dt.strftime('%Y')

        # Iterate through unique month names
        for month_name in df_gs['month_name'].unique():
            year = df_gs[df_gs['month_name'] == month_name]['year'].iloc[0]
            st.markdown(f'#### Events in {month_name} {year}')
            mon = df_gs[df_gs['month_name'] == month_name]
            df_mon = mon[['event_name', 'link', 'organiser', 'date_new', 'venue', 'details']]
            row_nu = len(df_mon.index)
            for i in range(row_nu):
                st.write(f"{i+1}) [{df_mon.iloc[i]['event_name']}]({df_mon.iloc[i]['link']}) organised by **{df_mon.iloc[i]['organiser']}**. Date: {df_mon.iloc[i]['date_new']}, Venue: {df_mon.iloc[i]['venue']}")
                if display:
                    st.caption(f"Details:\n{df_mon.iloc[i]['details']}")
        # df_gs.sort_values(by='date', ascending = True, inplace=True)
        # month_dict = {'01': 'January',
        #     '02': 'February',
        #     '03': 'March',
        #     '04': 'April',
        #     '05': 'May',
        #     '06': 'June',
        #     '07': 'July',
        #     '08': 'August',
        #     '09': 'September',
        #     '10': 'October',
        #     '11': 'November',
        #     '12': 'December'}
        # for month_num, month_name in month_dict.items():
        #     if month_num in df_gs['month'].values:
        #         st.markdown(f'#### Events in {month_name}')
        #         mon = df_gs[df_gs['month']==month_num] 
        #         df_mon = mon[['event_name', 'link', 'organiser', 'date_new', 'venue', 'details']]
        #         row_nu = len(df_mon.index)
        #         for i in range(row_nu):
        #             st.write(f"{i+1}) [{df_mon.iloc[i]['event_name']}]({df_mon.iloc[i]['link']}) organised by **{df_mon.iloc[i]['organiser']}**. Date: {df_mon.iloc[i]['date_new']}, Venue: {df_mon.iloc[i]['venue']}")
        #             if display:
        #                 st.caption(f"Details:\n{df_mon.iloc[i]['details']}")

    st.header('Past events')
    with st.expander('Expand to see the list'):
        years = df_gs2['year'].unique()[::-1]
        for year in years:
            if st.checkbox(f"Events in {year}", key=year):
                if year in df_gs2['year'].values:
                    events = df_gs2[df_gs2['year']==year].drop_duplicates(subset=['event_name', 'link', 'date'], keep='first')
                    events['link'] = events['link'].fillna('')
                    num_events = len(events.index)
                    event_strs = ('['+ events['event_name'] + ']'+ '('+ events['link'] + ')'', organised by ' + '**' + events['organiser'] + '**' + '. Date: ' + events['date_new'] + ', Venue: ' + events['venue'])
                    st.write(f"{num_events} events happened in {year}")
                    for i, event_str in enumerate(event_strs):
                        st.write(f"{i+1}) {event_str}")
    
    st.header('Event visuals')
    ap = ''
    ap2 = ''
    ap3 = ''
    selector = st.checkbox('Select a year')
    year = st.checkbox('Show years only')
    if selector:
        max_year = df_gs['date'].dt.year.max()
        min_year = df_gs['date'].dt.year.min()
        current_year = pd.Timestamp.now().year

        slider = st.slider('Select a year', 2024, max_year, current_year)
        slider = str(slider)
        df_gs_plot =df_gs_plot[df_gs_plot['year']==slider]
        ap = ' (in ' + slider+')'
    
    if year:
        date_plot=df_gs_plot['year'].value_counts()
        date_plot=date_plot.reset_index()
        date_plot=date_plot.rename(columns={'index':'Year','year':'Count'})
        date_plot.columns = ['Year', 'Count']
        date_plot=date_plot.sort_values(by='Year')
        fig = px.bar(date_plot, x='Year', y='Count')
        fig.update_xaxes(tickangle=-70)
        fig.update_layout(
            autosize=False,
            width=400,
            height=500)
        fig.update_layout(title={'text':'Events over time' +ap, 'y':0.95, 'x':0.5, 'yanchor':'top'})
        st.plotly_chart(fig, use_container_width = True)
    else:
        date_plot=df_gs_plot['month_year'].value_counts()
        date_plot=date_plot.reset_index()
        date_plot=date_plot.rename(columns={'index':'Date','month_year':'Count'})
        date_plot.columns = ['Date', 'Count']
        date_plot=date_plot.sort_values(by='Date')
        fig = px.bar(date_plot, x='Date', y='Count')
        fig.update_xaxes(tickangle=-70)
        fig.update_layout(
            autosize=False,
            width=400,
            height=500)
        fig.update_layout(title={'text':'Events over time' +ap, 'y':0.95, 'x':0.5, 'yanchor':'top'})
        st.plotly_chart(fig, use_container_width = True)

    organiser_plot = df_gs_plot['organiser'].value_counts()
    organiser_plot=organiser_plot.reset_index()
    organiser_plot=organiser_plot.rename(columns={'index':'Organiser', 'organiser':'Count'})
    organiser_plot.columns = ['Organiser','Count']
    organiser_plot=organiser_plot.sort_values(by='Count', ascending = False)
    organiser_plot_all=organiser_plot.copy()        
    all = st.checkbox('Show all organisers')
    if all:
        organiser_plot=organiser_plot_all
        ap2 = ' (all)'
    else:
        organiser_plot=organiser_plot.head(5)
        ap3 = ' (top 5) '
    fig = px.bar(organiser_plot, x='Organiser', y='Count', color='Organiser')
    fig.update_xaxes(tickangle=-65)
    fig.update_layout(
        autosize=False,
        width=400,
        height=700,
        showlegend=False)
    fig.update_layout(title={'text':'Events by organisers' + ap + ap2 +ap3, 'y':0.95, 'x':0.5, 'yanchor':'top'})
    st.plotly_chart(fig, use_container_width = True)
    with st.expander('See the list of event organisers'):
        row_nu_organiser= len(organiser_plot_all.index)
        organiser_plot_all=organiser_plot_all.sort_values('Organiser', ascending=True)
        for i in range(row_nu_organiser):
            st.caption(organiser_plot_all['Organiser'].iloc[i])


with tab2:
    st.subheader('Conferences')
    df_con = conn.read(spreadsheet='https://docs.google.com/spreadsheets/d/10ezNUOUpzBayqIMJWuS_zsvwklxP49zlfBWsiJI6aqI/edit#gid=939232836')
    df_con['date'] = pd.to_datetime(df_con['date'])
    df_con['date_new'] = df_con['date'].dt.strftime('%Y-%m-%d')
    df_con['date_new'] = pd.to_datetime(df_con['date'], dayfirst = True).dt.strftime('%d/%m/%Y')
    df_con['date_new_end'] = pd.to_datetime(df_con['date_end'], dayfirst = True).dt.strftime('%d/%m/%Y')
    df_con.sort_values(by='date', ascending = True, inplace=True)
    df_con['details'] = df_con['details'].fillna('No details')
    df_con['location'] = df_con['location'].fillna('No details')
    df_con = df_con.fillna('')
    df_con['date_end'] = pd.to_datetime(df_con['date'], dayfirst=True)    

    
    col1, col2 = st.columns(2)
    with col1:
        display = st.checkbox('Show details', key='conference')
    with col2:
        last_added = st.checkbox('Sort by most recently added', key='conference2')

    filter = df_con['date_end']>=pd.to_datetime(today)
    df_con = df_con.loc[filter]
    if df_con['conference_name'].any() in ("", [], None, 0, False):
        st.write('No upcoming conference!')

    if last_added:
        df_con = df_con.sort_index(ascending=False)
        df_con1 = ('['+ df_con['conference_name'] + ']'+ '('+ df_con['link'] + ')'', organised by ' + '**' + df_con['organiser'] + '**' + '. Date(s): ' + df_con['date_new'] + ' - ' + df_con['date_new_end'] + ', Venue: ' + df_con['venue'])
        row_nu = len(df_con.index)
        for i in range(row_nu):
            st.write(''+str(i+1)+') '+ df_con1.iloc[i])
            if display:
                st.caption('Conference place:'+'\n '+ df_con['location'].iloc[i])
                st.caption('Details:'+'\n '+ df_con['details'].iloc[i])

    else:
        df_con1 = ('['+ df_con['conference_name'] + ']'+ '('+ df_con['link'] + ')'', organised by ' + '**' + df_con['organiser'] + '**' + '. Date(s): ' + df_con['date_new'] + ' - ' + df_con['date_new_end'] + ', Venue: ' + df_con['venue'])
        row_nu = len(df_con.index)
        for i in range(row_nu):
            st.write(''+str(i+1)+') '+ df_con1.iloc[i])
            if display:
                st.caption('Conference place:'+'\n '+ df_con['location'].iloc[i])
                st.caption('Details:'+'\n '+ df_con['details'].iloc[i])
        
with tab3:
    st.subheader('Call for papers')
    df_cfp = conn.read(spreadsheet='https://docs.google.com/spreadsheets/d/10ezNUOUpzBayqIMJWuS_zsvwklxP49zlfBWsiJI6aqI/edit#gid=135096406') 

    df_cfp['deadline'] = pd.to_datetime(df_cfp['deadline'])
    df_cfp['deadline_new'] = df_cfp['deadline'].dt.strftime('%Y-%m-%d')
    df_cfp.sort_values(by='deadline', ascending = True, inplace=True)

    df_cfp['details'] = df_cfp['details'].fillna('No details')
    df_cfp = df_cfp.fillna('')

    df_cfp = df_cfp.drop_duplicates(subset=['name', 'link', 'deadline'], keep='first')
    
    display = st.checkbox('Show details', key='cfp')

    df_cfp['deadline'] = pd.to_datetime(df_cfp['deadline'], dayfirst=True)

    filter = df_cfp['deadline']>=pd.to_datetime(today)
    df_cfp = df_cfp.loc[filter]
    if df_cfp['name'].any() in ("", [], None, 0, False):
        st.write('No upcoming Call for papers!')

    df_cfp1 = ('['+ df_cfp['name'] + ']'+ '('+ df_cfp['link'] + ')'', organised by ' + '**' + df_cfp['organiser'] + '**' + '. Deadline: ' + df_cfp['deadline_new'])
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
© 2024 All rights reserved. This website is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">Creative Commons Attribution 4.0 International License</a>.
"""
)
