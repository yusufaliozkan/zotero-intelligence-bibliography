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


st.set_page_config(layout = "wide", 
                    page_title='Intelligence studies network',
                    page_icon="https://images.pexels.com/photos/315918/pexels-photo-315918.png",
                    initial_sidebar_state="auto") 

st.title("Intelligence studies network")
st.header('Resources on intelligence studies')
st.info('This page lists institutions, academic programs, and  other resources on intelligence studies!!')


image = 'https://images.pexels.com/photos/315918/pexels-photo-315918.png'

with st.sidebar:

    st.image(image, width=150)
    st.sidebar.markdown("# Intelligence studies network")
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
    with st.expander('Disclaimer and acknowledgements'):
        st.warning('''
        This website and the Intelligence bibliography Zotero group library do not list all the sources on intelligence studies. 
        The list is created based on the creator's subjective views.
        ''')
        st.info('''
        The following sources are used to collate some of the articles and events: [KISG digest](https://kisg.co.uk/kisg-digests), [IAFIE digest compiled by Filip Kovacevic](https://www.iafie.org/Login.aspx)
        ''')
    with st.expander('Contact us'):
        st.write('If you have any questions or suggestions, please do get in touch with us by filling the form [here](https://www.intelligencenetwork.org/contact-us).')
        st.write('Report your technical issues or requests [here](https://github.com/YusufAliOzkan/zotero-intelligence-bibliography/issues).')


col1, col2 = st.columns([5,2])
with col1:
    conn = connect()

    # Perform SQL query on the Google Sheet.
    # Uses st.cache to only rerun when the query changes or after 10 min.
    @st.cache_resource(ttl=10)
    def run_query(query):
        rows = conn.execute(query, headers=1)
        rows = rows.fetchall()
        return rows

    sheet_url = st.secrets["public_gsheets_url_orgs"]
    rows = run_query(f'SELECT * FROM "{sheet_url}"')

    data = []
    columns = ['Type', 'Institution', 'Programme_level', 'Programme_name', 'Link', 'Country', 'Status']

    # Print results.
    for row in rows:
        data.append((row.Type, row.Institution, row.Programme_level, row.Programme_name, row.Link, row.Country, row.Status))
    df = pd.DataFrame(data, columns=columns)
    countries = df['Country'].unique()
    types = df['Type'].unique()

    df = df[df['Status'] == 'Active']
    df = df.sort_values(by='Institution')

    def display_numbered_list(programs, column_name, show_country=False):
        counter = 1
        for index, row in programs.iterrows():
            programme_level = row['Programme_level']
            programme_name = row['Programme_name']
            
            programme_info = ""
            if programme_level and programme_name:
                programme_info = f"{programme_level}: [{programme_name}]({row['Link']}), *{row['Institution']}*, {row['Country']}"
            else:
                programme_info = f"{programme_level}: [{row['Institution']}]({row['Link']}), {row['Country']}"
            
            if show_country:
                programme_info += f", {row['Country']}"
            
            st.write(f"{counter}. {programme_info}")
            counter += 1

    for prog_type in types:
        type_programs = df[df['Type'] == prog_type]
        num_unique_countries = type_programs['Country'].nunique()

        with st.expander(f"{prog_type} ({len(type_programs)})"):
            if prog_type == 'Academic programs':
                country_counts = type_programs['Country'].value_counts().sort_values(ascending=False)
                countries_sorted = country_counts.index.tolist()
                country_counts_dict = {country: f"{country} ({count})" for country, count in country_counts.items()}

                selected_country = st.multiselect('Filter by country:', countries_sorted, format_func=lambda x: country_counts_dict[x])
                
                if selected_country:
                    type_programs = type_programs[type_programs['Country'].isin(selected_country)]
                    programme_levels = type_programs['Programme_level'].unique()
                
                programme_levels = type_programs['Programme_level'].unique()
                selected_level = st.selectbox("Filter by Programme Level:", ['All'] + list(programme_levels))

                if selected_level != 'All':
                    type_programs = type_programs[type_programs['Programme_level'] == selected_level]

                num_unique_countries = type_programs['Country'].nunique()
                if num_unique_countries==1:
                    selected_country_str = selected_country[0].split(" (")[0]
                    st.write(f'**{len(type_programs)} program(s) found in {selected_country_str}**')
                else:
                    st.write(f'**{len(type_programs)} program(s) found in {num_unique_countries} countries**')

            if prog_type != 'Academic programs':
                if num_unique_countries!=1:
                    num_unique_countries = type_programs['Country'].nunique()
                    country_counts = type_programs['Country'].value_counts().sort_values(ascending=False)
                    countries_sorted = country_counts.index.tolist()
                    country_counts_dict = {country: f"{country} ({count})" for country, count in country_counts.items()}
                    selected_country = st.multiselect('Filter by country:', countries_sorted, format_func=lambda x: country_counts_dict[x])
                    if selected_country:
                        type_programs = type_programs[type_programs['Country'].isin(selected_country)]
                        num_unique_countries = type_programs['Country'].nunique()
                        if num_unique_countries==1:
                            selected_country_str = selected_country[0].split(" (")[0]
                            st.write(f'**{len(type_programs)} {prog_type} found in {selected_country_str}**')
                        else:
                            st.write(f'**{len(type_programs)} {prog_type} found in {num_unique_countries} countries**')
                    else:
                        st.write(f'**{len(type_programs)} {prog_type} found in {num_unique_countries} countries**')
            display_numbered_list(type_programs, prog_type, show_country=False if prog_type != 'Academic' else False)

with col2:
    with st.expander('Collections', expanded=True):
        st.caption('[Intelligence history](https://intelligence.streamlit.app/Intelligence_history)')
        st.caption('[Intelligence studies](https://intelligence.streamlit.app/Intelligence_studies)')
        st.caption('[Intelligence analysis](https://intelligence.streamlit.app/Intelligence_analysis)')
        st.caption('[Intelligence organisations](https://intelligence.streamlit.app/Intelligence_organisations)')
        st.caption('[Intelligence failures](https://intelligence.streamlit.app/Intelligence_failures)')
        st.caption('[Intelligence oversight and ethics](https://intelligence.streamlit.app/Intelligence_oversight_and_ethics)')
        st.caption('[Intelligence collection](https://intelligence.streamlit.app/Intelligence_collection)')
        st.caption('[Counterintelligence](https://intelligence.streamlit.app/Counterintelligence)')
        st.caption('[Covert action](https://intelligence.streamlit.app/Covert_action)')
        st.caption('[Intelligence and cybersphere](https://intelligence.streamlit.app/Intelligence_and_cybersphere)')
        st.caption('[Global intelligence](https://intelligence.streamlit.app/Global_intelligence)')
        st.caption('[AI and intelligence](https://intelligence.streamlit.app/AI_and_intelligence)')
        st.caption('[Special collections](https://intelligence.streamlit.app/Special_collections)')

    with st.expander('Events & conferences', expanded=True):
        st.markdown('##### Next event')
        conn = connect()

        # Perform SQL query on the Google Sheet.
        # Uses st.cache to only rerun when the query changes or after 10 min.
        @st.cache_resource(ttl=10)
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

        sheet_url_forms = st.secrets["public_gsheets_url_forms"]
        rows = run_query(f'SELECT * FROM "{sheet_url_forms}"')
        data = []
        columns = ['event_name', 'organiser', 'link', 'date', 'venue', 'details']
        # Print results.
        for row in rows:
            data.append((row.Event_name, row.Event_organiser, row.Link_to_the_event, row.Date_of_event, row.Event_venue, row.Details))
        pd.set_option('display.max_colwidth', None)
        df_forms = pd.DataFrame(data, columns=columns)

        df_forms['date_new'] = pd.to_datetime(df_forms['date'], dayfirst = True).dt.strftime('%d/%m/%Y')
        df_forms['month'] = pd.to_datetime(df_forms['date'], dayfirst = True).dt.strftime('%m')
        df_forms['year'] = pd.to_datetime(df_forms['date'], dayfirst = True).dt.strftime('%Y')
        df_forms['month_year'] = pd.to_datetime(df_forms['date'], dayfirst = True).dt.strftime('%Y-%m')
        df_forms.sort_values(by='date', ascending = True, inplace=True)
        df_forms = df_forms.drop_duplicates(subset=['event_name', 'link', 'date'], keep='first')
        
        df_forms['details'] = df_forms['details'].fillna('No details')
        df_forms = df_forms.fillna('')
        df_gs = pd.concat([df_gs, df_forms], axis=0)
        df_gs = df_gs.reset_index(drop=True)
        df_gs = df_gs.drop_duplicates(subset=['event_name', 'link', 'date'], keep='first')

        df_gs.sort_values(by='date', ascending = True, inplace=True)
        df_gs = df_gs.drop_duplicates(subset=['event_name', 'link'], keep='first')
        df_gs = df_gs.fillna('')
        today = dt.date.today()
        filter = (df_gs['date']>=today)
        df_gs = df_gs.loc[filter]
        df_gs = df_gs.head(1)
        if df_gs['event_name'].any() in ("", [], None, 0, False):
            st.write('No upcoming event!')
        df_gs1 = ('['+ df_gs['event_name'] + ']'+ '('+ df_gs['link'] + ')'', organised by ' + '**' + df_gs['organiser'] + '**' + '. Date: ' + df_gs['date_new'] + ', Venue: ' + df_gs['venue'])
        row_nu = len(df_gs.index)
        for i in range(row_nu):
            st.write(df_gs1.iloc[i])
        
        st.markdown('##### Next conference')
        sheet_url2 = st.secrets["public_gsheets_url2"]
        rows = run_query(f'SELECT * FROM "{sheet_url2}"')
        data = []
        columns = ['conference_name', 'organiser', 'link', 'date', 'date_end', 'venue', 'details', 'location']
        for row in rows:
            data.append((row.conference_name, row.organiser, row.link, row.date, row.date_end, row.venue, row.details, row.location))
        pd.set_option('display.max_colwidth', None)
        df_con = pd.DataFrame(data, columns=columns)
        df_con['date_new'] = pd.to_datetime(df_con['date'], dayfirst = True).dt.strftime('%d/%m/%Y')
        df_con['date_new_end'] = pd.to_datetime(df_con['date_end'], dayfirst = True).dt.strftime('%d/%m/%Y')
        df_con.sort_values(by='date', ascending = True, inplace=True)
        df_con['details'] = df_con['details'].fillna('No details')
        df_con['location'] = df_con['location'].fillna('No details')
        df_con = df_con.fillna('')            
        filter = (df_con['date_end']>=today)
        df_con = df_con.loc[filter]
        df_con = df_con.head(1)
        if df_con['conference_name'].any() in ("", [], None, 0, False):
            st.write('No upcoming conference!')
        df_con1 = ('['+ df_con['conference_name'] + ']'+ '('+ df_con['link'] + ')'', organised by ' + '**' + df_con['organiser'] + '**' + '. Date(s): ' + df_con['date_new'] + ' - ' + df_con['date_new_end'] + ', Venue: ' + df_con['venue'])
        row_nu = len(df_con.index)
        for i in range(row_nu):
            st.write( df_con1.iloc[i])
        st.write('Visit the [Events on intelligence](https://intelligence.streamlit.app/Events) page to see more!')

    with st.expander('Digest', expanded=True):
        st.write('See our dynamic [digest](https://intelligence.streamlit.app/Digest) for the latest updates on intelligence!')
st.write('---')

components.html(
"""
<a rel="license" href="http://creativecommons.org/licenses/by/4.0/"><img alt="Creative Commons Licence" style="border-width:0" 
src="https://i.creativecommons.org/l/by/4.0/80x15.png" /></a><br />
© 2022 All rights reserved. This website is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">Creative Commons Attribution 4.0 International License</a>.
"""
)
