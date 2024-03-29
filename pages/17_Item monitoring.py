# from pyzotero import zotero
import pandas as pd
import streamlit as st
from IPython.display import HTML
import streamlit.components.v1 as components
# import numpy as np
# import altair as alt
# from pandas.io.json import json_normalize
from datetime import date, timedelta  
from datetime import datetime
import datetime 
import datetime as dt
# import plotly.express as px
# import numpy as np
# import re
from gsheetsdb import connect
# from fpdf import FPDF
# import base64
from sidebar_content import sidebar_content
import requests
from rss_feed import df_podcast, df_magazines


st.set_page_config(layout = "wide", 
                    page_title='Intelligence studies network',
                    page_icon="https://images.pexels.com/photos/315918/pexels-photo-315918.png",
                    initial_sidebar_state="auto") 

st.title("Intelligence studies network")
st.header('Item monitoring')
st.info('''
        The following items are not in the library yet. Book reviews will not be included!
        ''')


image = 'https://images.pexels.com/photos/315918/pexels-photo-315918.png'

with st.sidebar:

    sidebar_content()


col1, col2 = st.columns([5,2])
with col1:

    item_monitoring = st.button("Item monitoring")
    if item_monitoring:
        st.subheader('Monitoring section')
        st.write('The following items are not in the library yet. Book reviews will not be included!')
        with st.status("Scanning sources to find items...", expanded=True) as status:
        # with st.spinner('Scanning sources to find items...'): 
            api_links = [
                'https://api.openalex.org/works?filter=primary_location.source.id:s33269604&sort=publication_year:desc&per_page=10', #IJIC
                "https://api.openalex.org/works?filter=primary_location.source.id:s205284143&sort=publication_year:desc&per_page=10", #The Historical Journal
                'https://api.openalex.org/works?filter=primary_location.source.id:s4210168073&sort=publication_year:desc&per_page=10', #INS
                'https://api.openalex.org/works?filter=primary_location.source.id:s2764506647&sort=publication_year:desc&per_page=10', #Journal of Intelligence History
                'https://api.openalex.org/works?filter=primary_location.source.id:s2764781490&sort=publication_year:desc&per_page=10', #Journal of Policing, Intelligence and Counter Terrorism
                'https://api.openalex.org/works?filter=primary_location.source.id:s93928036&sort=publication_year:desc&per_page=10', #Cold War History
                'https://api.openalex.org/works?filter=primary_location.source.id:s962698607&sort=publication_year:desc&per_page=10', #RUSI Journal
                'https://api.openalex.org/works?filter=primary_location.source.id:s199078552&sort=publication_year:desc&per_page=10', #Journal of Strategic Studies
                'https://api.openalex.org/works?filter=primary_location.source.id:s145781505&sort=publication_year:desc&per_page=10', #War in History
                'https://api.openalex.org/works?filter=primary_location.source.id:s120387555&sort=publication_year:desc&per_page=10', #International History Review
                'https://api.openalex.org/works?filter=primary_location.source.id:s161550498&sort=publication_year:desc&per_page=10', #Journal of Contemporary History
                'https://api.openalex.org/works?filter=primary_location.source.id:s164505828&sort=publication_year:desc&per_page=10', #Middle Eastern Studies
                'https://api.openalex.org/works?filter=primary_location.source.id:s99133842&sort=publication_year:desc&per_page=10', #Diplomacy & Statecraft
                'https://api.openalex.org/works?filter=primary_location.source.id:s4210219209&sort=publication_year:desc&per_page=10', #The international journal of intelligence, security, and public affair
                'https://api.openalex.org/works?filter=primary_location.source.id:s185196701&sort=publication_year:desc&per_page=10',#Cryptologia
                'https://api.openalex.org/works?filter=primary_location.source.id:s157188123&sort=publication_year:desc&per_page=10', #The Journal of Slavic Military Studies
                'https://api.openalex.org/works?filter=primary_location.source.id:s79519963&sort=publication_year:desc&per_page=10',#International Affairs
                'https://api.openalex.org/works?filter=primary_location.source.id:s161027966&sort=publication_year:desc&per_page=10', #Political Science Quarterly
                'https://api.openalex.org/works?filter=primary_location.source.id:s4210201145&sort=publication_year:desc&per_page=10', #Journal of intelligence, conflict and warfare
                'https://api.openalex.org/works?filter=primary_location.source.id:s2764954702&sort=publication_year:desc&per_page=10', #The Journal of Conflict Studies
                'https://api.openalex.org/works?filter=primary_location.source.id:s200077084&sort=publication_year:desc&per_page=10', #Journal of Cold War Studies
                'https://api.openalex.org/works?filter=primary_location.source.id:s27717133&sort=publication_year:desc&per_page=10', #Survival
                'https://api.openalex.org/works?filter=primary_location.source.id:s4210214688&sort=publication_year:desc&per_page=10', #Security and Defence Quarterly
                'https://api.openalex.org/works?filter=primary_location.source.id:s112911512&sort=publication_year:desc&per_page=10', #The Journal of Imperial and Commonwealth History
                'https://api.openalex.org/works?filter=primary_location.source.id:s131264395&sort=publication_year:desc&per_page=10', #Review of International Studies
                'https://api.openalex.org/works?filter=primary_location.source.id:s154084123&sort=publication_year:desc&per_page=10', #Diplomatic History
                'https://api.openalex.org/works?filter=primary_location.source.id:s103350616&sort=publication_year:desc&per_page=10', #Cambridge Review of International Affairs
                'https://api.openalex.org/works?filter=primary_location.source.id:s17185278&sort=publication_year:desc&per_page=10', #Public Policy and Administration
                'https://api.openalex.org/works?filter=primary_location.source.id:s21016770&sort=publication_year:desc&per_page=10', #Armed Forces & Society
                'https://api.openalex.org/works?filter=primary_location.source.id:s41746314&sort=publication_year:desc&per_page=10', #Studies in Conflict & Terrorism
                'https://api.openalex.org/works?filter=primary_location.source.id:s56601287&sort=publication_year:desc&per_page=10', #The English Historical Review
                'https://api.openalex.org/works?filter=primary_location.source.id:s143110675&sort=publication_year:desc&per_page=10', #World Politics
                'https://api.openalex.org/works?filter=primary_location.source.id:s106532728&sort=publication_year:desc&per_page=10', #Israel Affairs
                'https://api.openalex.org/works?filter=primary_location.source.id:s67329160&sort=publication_year:desc&per_page=10', #Australian Journal of International Affairs
                'https://api.openalex.org/works?filter=primary_location.source.id:s49917718&sort=publication_year:desc&per_page=10', #Contemporary British History
                'https://api.openalex.org/works?filter=primary_location.source.id:s8593340&sort=publication_year:desc&per_page=10', #The Historian
                'https://api.openalex.org/works?filter=primary_location.source.id:s161552967&sort=publication_year:desc&per_page=10', #The British Journal of Politics and International Relations
                'https://api.openalex.org/works?filter=primary_location.source.id:s141724154&sort=publication_year:desc&per_page=10', #Terrorism and Political Violence
                'https://api.openalex.org/works?filter=primary_location.source.id:s53578506&sort=publication_year:desc&per_page=10', #Mariner's Mirror
                'https://api.openalex.org/works?filter=primary_location.source.id:s4210184262&sort=publication_year:desc&per_page=10', #Small Wars & Insurgencies
                'https://api.openalex.org/works?filter=primary_location.source.id:s4210236978&sort=publication_year:desc&per_page=10', #Journal of Cyber Policy
                'https://api.openalex.org/works?filter=primary_location.source.id:s120889147&sort=publication_year:desc&per_page=10', #South Asia:Journal of South Asian Studies
                'https://api.openalex.org/works?filter=primary_location.source.id:s86954274&sort=cited_by_count:desc&per_page=10', #International Journal

                # Add more API links here
            ]

            # Define journals to include filtered items
            journals_with_filtered_items = [
                'The Historical Journal', 'Journal of Policing, Intelligence and Counter Terrorism', 'Cold War History', 'RUSI Journal',
                'Journal of Strategic Studies', 'War in History', 'International History Review','Journal of Contemporary History', 
                'Middle Eastern Studies', 'Diplomacy & Statecraft', 'The international journal of intelligence, security, and public affairs',
                'Cryptologia', 'The Journal of Slavic Military Studies', 'International Affairs', 'Political Science Quarterly',
                'Journal of intelligence, conflict and warfare', 'The Journal of Conflict Studies','Journal of Cold War Studies', 'Survival',
                'Security and Defence Quarterly', 'The Journal of Imperial and Commonwealth History', 'Review of International Studies', 'Diplomatic History',
                'Cambridge Review of International Affairs', 'Public Policy and Administration', 'Armed Forces & Society', 'Studies in Conflict & Terrorism',
                'The English Historical Review', 'World Politics', 'Israel Affairs', 'Australian Journal of International Affairs', 'Contemporary British History',
                'The Historian', 'The British Journal of Politics and International Relations', 'Terrorism and Political Violence', "Mariner's Mirror",
                'Small Wars & Insurgencies', 'Journal of Cyber Policy', 'South Asia:Journal of South Asian Studies', 'International Journal'
                ]

            # Define keywords for filtering
            keywords = [
                'intelligence', 'spy', 'counterintelligence', 'espionage', 'covert', 'signal', 'sigint', 'humint', 'decipher', 'cryptanalysis',
                'spying', 'spies'
                ] 

            # Initialize an empty list to store DataFrame for each API link
            dfs = []

            # Loop through each API link
            for api_link in api_links:
                # Send a GET request to the API
                response = requests.get(api_link)

                # Check if the request was successful
                if response.status_code == 200:
                    # Parse the JSON response
                    data = response.json()
                    
                    # Extract the results
                    results = data['results']
                    
                    # Initialize lists to store values for DataFrame
                    titles = []
                    dois = []
                    publication_dates = []
                    dois_without_https = []
                    journals = []
                    
                    # Get today's date
                    today = datetime.datetime.today().date()
                    
                    # Extract data for each result
                    for result in results:
                        # Convert publication date string to datetime object
                        pub_date = datetime.datetime.strptime(result['publication_date'], '%Y-%m-%d').date()
                        
                        # Check if the publication date is within the last # days
                        if today - pub_date <= timedelta(days=90):
                            titles.append(result['title'])
                            dois.append(result['doi'])
                            publication_dates.append(result['publication_date'])
                            dois_without_https.append(result['ids']['doi'].split("https://doi.org/")[-1])
                            journals.append(result['primary_location']['source']['display_name'])
                    
                    # Create DataFrame
                    df = pd.DataFrame({
                        'Title': titles,
                        'Link': dois,
                        'Publication Date': publication_dates,
                        'DOI': dois_without_https,
                        'Journal': journals
                    })
                    
                    # Append DataFrame to the list
                    dfs.append(df)
                
                else:
                    print(f"Failed to fetch data from the API: {api_link}")

            # Concatenate DataFrames from all API links
            final_df = pd.concat(dfs, ignore_index=True)

            # Filter 'The Historical Journal' to only include titles containing keywords
            historical_journal_filtered = final_df[final_df['Journal'].isin(journals_with_filtered_items)]
            historical_journal_filtered = historical_journal_filtered[historical_journal_filtered['Title'].str.lower().str.contains('|'.join(keywords))]

            # Filter other journals to exclude 'The Historical Journal'
            other_journals = final_df[~final_df['Journal'].isin(journals_with_filtered_items)]

            # Concatenate the filtered DataFrames
            filtered_final_df = pd.concat([other_journals, historical_journal_filtered], ignore_index=True)

            df_dedup = pd.read_csv('all_items.csv')
            df_dois = df_dedup.copy() 
            df_dois.dropna(subset=['DOI'], inplace=True) 
            column_to_keep = 'DOI'
            df_dois = df_dois[[column_to_keep]]
            df_dois = df_dois.reset_index(drop=True) 

            merged_df = pd.merge(filtered_final_df, df_dois[['DOI']], on='DOI', how='left', indicator=True)
            items_not_in_df2 = merged_df[merged_df['_merge'] == 'left_only']
            items_not_in_df2.drop('_merge', axis=1, inplace=True)

            words_to_exclude = ['notwantedwordshere'] #'paperback', 'hardback']

            mask = ~items_not_in_df2['Title'].str.contains('|'.join(words_to_exclude), case=False)
            items_not_in_df2 = items_not_in_df2[mask]
            items_not_in_df2 = items_not_in_df2.reset_index(drop=True)
            st.write('**Journal articles**')
            row_nu = len(items_not_in_df2.index)
            if row_nu == 0:
                st.write('No new podcast published!')
            else:
                items_not_in_df2 = items_not_in_df2.sort_values(by=['Publication Date'], ascending=False).reset_index(drop=True)
                items_not_in_df2

            df_item_podcast = df_dedup.copy()
            df_item_podcast.dropna(subset=['Title'], inplace=True)
            column_to_keep = 'Title'
            df_item_podcast = df_item_podcast[[column_to_keep]]
            df_podcast = pd.merge(df_podcast, df_item_podcast[['Title']], on='Title', how='left', indicator=True)
            items_not_in_df_item_podcast = df_podcast[df_podcast['_merge'] == 'left_only']
            items_not_in_df_item_podcast.drop('_merge', axis=1, inplace=True)
            items_not_in_df_item_podcast = items_not_in_df_item_podcast.reset_index(drop=True)
            st.write('**Podcasts**')
            row_nu = len(items_not_in_df_item_podcast.index)
            if row_nu == 0:
                st.write('No new podcast published!')
            else:
                items_not_in_df_item_podcast = items_not_in_df_item_podcast.sort_values(by=['PubDate'], ascending=False)
                items_not_in_df_item_podcast

            df_item_magazines = df_dedup.copy()
            df_item_magazines.dropna(subset=['Title'], inplace=True)
            column_to_keep = 'Title'
            df_item_magazines = df_item_magazines[[column_to_keep]]
            df_magazines = pd.merge(df_magazines, df_item_magazines[['Title']], on='Title', how='left', indicator=True)
            items_not_in_df_item_magazines = df_magazines[df_magazines['_merge'] == 'left_only']
            items_not_in_df_item_magazines.drop('_merge', axis=1, inplace=True)
            items_not_in_df_item_magazines = items_not_in_df_item_magazines.reset_index(drop=True)
            st.write('**Magazine articles**')
            row_nu = len(items_not_in_df_item_magazines.index)
            if row_nu == 0:
                st.write('No new magazine article published!')
            else:
                items_not_in_df_item_magazines = items_not_in_df_item_magazines.sort_values(by=['PubDate'], ascending=False)
                items_not_in_df_item_magazines        
            status.update(label="Search complete!", state="complete", expanded=True)

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
© 2024 Yusuf Ozkan. All rights reserved. This website is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">Creative Commons Attribution 4.0 International License</a>.
"""
)
