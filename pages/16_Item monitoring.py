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
# from fpdf import FPDF
# import base64
from sidebar_content import sidebar_content
import requests
from rss_feed import df_podcast, df_magazines
from events import evens_conferences
import xml.etree.ElementTree as ET
from fuzzywuzzy import fuzz

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
                'https://api.openalex.org/works?filter=primary_location.source.id:s86954274&sort=publication_year:desc&per_page=10', #International Journal
                'https://api.openalex.org/works?page=1&filter=primary_location.source.id:s117224066&sort=publication_year:desc', #German Law Journal
                'https://api.openalex.org/works?page=1&filter=primary_location.source.id:s160097506&sort=publication_year:desc', #American Journal of International Law
                'https://api.openalex.org/works?page=1&filter=primary_location.source.id:s175405714&sort=publication_year:desc', #European Journal of International Law
                'https://api.openalex.org/works?page=1&filter=primary_location.source.id:s84944781&sort=publication_year:desc', #Human Rights Law Review
                'https://api.openalex.org/works?page=1&filter=primary_location.source.id:s154337186&sort=publication_year:desc', #Leiden Journal of International Law
                'https://api.openalex.org/works?page=1&filter=primary_location.source.id:s156235965&sort=publication_year:desc', #International & Comparative Law Quarterl
                'https://api.openalex.org/works?page=1&filter=primary_location.source.id:s68909633&sort=publication_year:desc', #Journal of Conflict and Security Law
                'https://api.openalex.org/works?page=1&filter=primary_location.source.id:s42104779&sort=publication_year:desc', #Journal of International Dispute Settlement
                'https://api.openalex.org/works?page=1&filter=primary_location.source.id:s2764513295&sort=publication_year:desc', #Security and Human Rights
                'https://api.openalex.org/works?page=1&filter=primary_location.source.id:s82119083&sort=publication_year:desc', #Modern Law Review
                'https://api.openalex.org/works?page=1&filter=primary_location.source.id:s129176075&sort=publication_year:desc', #International Theory
                'https://api.openalex.org/works?page=1&filter=primary_location.source.id:s2764608241&sort=publication_year:desc', #Michigan Journal of International Law
                'https://api.openalex.org/works?page=1&filter=primary_location.source.id:s2735957470&sort=publication_year:desc', #Journal of Global Security Studies
                'https://api.openalex.org/works?page=1&filter=primary_topic.id:t12572&sort=publication_year:desc', #Intelligence Studies and Analysis in Modern Context
                'https://api.openalex.org/works?page=1&filter=concepts.id:c558872910&sort=publication_year:desc', #ConceptEspionage
                'https://api.openalex.org/works?page=1&filter=concepts.id:c173127888&sort=publication_year:desc', #ConceptCounterintelligence

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
                'Small Wars & Insurgencies', 'Journal of Cyber Policy', 'South Asia:Journal of South Asian Studies', 'International Journal', 'German Law Journal',
                'American Journal of International Law', 'European Journal of International Law', 'Human Rights Law Review', 'Leiden Journal of International Law',
                'International & Comparative Law Quarterly', 'Journal of Conflict and Security Law', 'Journal of International Dispute Settlement', 'Security and Human Rights',
                'Modern Law Review', 'International Theory', 'Michigan Journal of International Law', 'Journal of Global Security Studies', 'Intelligence Studies and Analysis in Modern Context'
                ]

            # Define keywords for filtering
            keywords = [
                'intelligence', 'spy', 'counterintelligence', 'espionage', 'covert', 'signal', 'sigint', 'humint', 'decipher', 'cryptanalysis',
                'spying', 'spies', 'surveillance', 'targeted killing', 'cyberespionage', ' cia ', 'rendition', ' mi6 ', ' mi5 ', ' sis ', 'security service',
                'central intelligence'
            ]

            dfs = []

            for api_link in api_links:
                response = requests.get(api_link)

                if response.status_code == 200:
                    data = response.json()
                    results = data.get('results', [])

                    titles = []
                    dois = []
                    publication_dates = []
                    dois_without_https = []
                    journals = []

                    today = datetime.datetime.today().date()

                    for result in results:
                        if result is None:
                            continue
                        
                        pub_date_str = result.get('publication_date')
                        if pub_date_str is None:
                            continue

                        try:
                            pub_date = datetime.datetime.strptime(pub_date_str, '%Y-%m-%d').date()
                        except ValueError:
                            continue  # Skip this result if the date is not in the expected format

                        if today - pub_date <= timedelta(days=90):
                            title = result.get('title')
                            
                            if title is not None and any(keyword in title.lower() for keyword in keywords):
                                titles.append(title)
                                dois.append(result.get('doi', 'Unknown'))
                                publication_dates.append(pub_date_str)
                                
                                # Ensure 'ids' and 'doi' are present before splitting
                                ids = result.get('ids', {})
                                doi_value = ids.get('doi', 'Unknown')
                                if doi_value != 'Unknown':
                                    dois_without_https.append(doi_value.split("https://doi.org/")[-1])
                                else:
                                    dois_without_https.append('Unknown')

                                # Safely navigate through nested dictionaries using get
                                primary_location = result.get('primary_location', {})
                                source = primary_location.get('source')
                                if source:
                                    journal_name = source.get('display_name', 'Unknown')
                                else:
                                    journal_name = 'Unknown'

                                journals.append(journal_name)

                    if titles:  # Ensure DataFrame creation only if there are titles
                        df = pd.DataFrame({
                            'Title': titles,
                            'Link': dois,
                            'Publication Date': publication_dates,
                            'DOI': dois_without_https,
                            'Journal': journals,
                        })

                        dfs.append(df)

            # Combine all DataFrames in dfs list into a single DataFrame
            if dfs:
                final_df = pd.concat(dfs, ignore_index=True)
            else:
                final_df = pd.DataFrame()  # Create an empty DataFrame if dfs is empty

                # else:
                #     print(f"Failed to fetch data from the API: {api_link}")

            final_df = pd.concat(dfs, ignore_index=True)
            final_df = final_df.drop_duplicates(subset='Link')

            historical_journal_filtered = final_df[final_df['Journal'].isin(journals_with_filtered_items)]

            other_journals = final_df[~final_df['Journal'].isin(journals_with_filtered_items)]

            filtered_final_df = pd.concat([other_journals, historical_journal_filtered], ignore_index=True)

            ## DOI based filtering
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
            st.write('**Journal articles (DOI based filtering)**')
            row_nu = len(items_not_in_df2.index)
            if row_nu == 0:
                st.write('No new podcast published!')
            else:
                items_not_in_df2 = items_not_in_df2.sort_values(by=['Publication Date'], ascending=False)
                items_not_in_df2 = items_not_in_df2.reset_index(drop=True)
                items_not_in_df2

            ## Title based filtering
            df_titles = df_dedup.copy()
            df_titles.dropna(subset=['Title'], inplace=True)
            column_to_keep = 'Title'
            df_titles = df_titles[[column_to_keep]]
            df_titles = df_titles.reset_index(drop=True)

            merged_df_2 = pd.merge(items_not_in_df2, df_titles[['Title']], on='Title', how='left', indicator=True)
            items_not_in_df3 = merged_df_2[merged_df_2['_merge'] == 'left_only']
            items_not_in_df3.drop('_merge', axis=1, inplace=True)
            items_not_in_df3 = items_not_in_df3.sort_values(by=['Publication Date'], ascending=False)
            items_not_in_df3 = items_not_in_df3.reset_index(drop=True)


            st.write('**Journal articles (future publications)**')
            ## Future publications
            items_not_in_df2['Publication Date'] = pd.to_datetime(items_not_in_df2['Publication Date'])
            current_date = datetime.datetime.now()
            future_df = items_not_in_df2[items_not_in_df2['Publication Date']>=current_date]
            future_df = future_df.reset_index(drop=True)
            future_df

            ## Published in the last 30 days
            st.write('**Journal articles (published in last 30 days)**')
            current_date = datetime.datetime.now()
            date_30_days_ago = current_date - timedelta(days=30)
            last_30_days_df = items_not_in_df2[(items_not_in_df2['Publication Date']<=current_date) & (items_not_in_df2['Publication Date']>=date_30_days_ago)]
            last_30_days_df = last_30_days_df.reset_index(drop=True)
            last_30_days_df

            # merged_df = pd.merge(filtered_final_df, df_dois[['DOI']], on='DOI', how='left', indicator=True)
            # items_not_in_df2 = merged_df[merged_df['_merge'] == 'left_only']
            # items_not_in_df2.drop('_merge', axis=1, inplace=True)

            # words_to_exclude = ['notwantedwordshere'] #'paperback', 'hardback']

            # mask = ~items_not_in_df2['Title'].str.contains('|'.join(words_to_exclude), case=False)
            # items_not_in_df2 = items_not_in_df2[mask]
            # items_not_in_df2 = items_not_in_df2.reset_index(drop=True)
            # st.write('**Journal articles**')
            # row_nu = len(items_not_in_df2.index)
            # if row_nu == 0:
            #     st.write('No new podcast published!')
            # else:
            #     items_not_in_df2 = items_not_in_df2.sort_values(by=['Publication Date'], ascending=False)
            #     items_not_in_df2 = items_not_in_df2.reset_index(drop=True)
            #     items_not_in_df2

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

            st.write('**Institutional repositories**')

            def fetch_rss_data(url, label):
                response = requests.get(url)
                rss_content = response.content
                root = ET.fromstring(rss_content)
                items = root.findall('.//item')[1:]
                data = []
                for item in items:
                    title = item.find('title').text
                    link = item.find('link').text
                    data.append({'title': title, 'link': link, 'label': label})
                return data

            # URLs of the RSS feeds with their respective labels
            feeds = [
                {"url": "https://rss.app/feeds/uBBTAmA7a9rMr7JA.xml", "label": "Brunel University"},
                {"url": "https://rss.app/feeds/S566whCCjTbiXmns.xml", "label": "Leiden University"}
            ]

            # Fetch and combine data from both RSS feeds
            all_data = []
            for feed in feeds:
                all_data.extend(fetch_rss_data(feed["url"], feed["label"]))

            # Create a DataFrame
            df = pd.DataFrame(all_data)
            words_to_filter = ["intelligence", "espionage", "spy", "oversight"]
            pattern = '|'.join(words_to_filter)

            df = df[df['title'].str.contains(pattern, case=False, na=False)].reset_index(drop=True)
            df['title'] = df['title'].str.replace('Brunel University Research Archive:', '', regex=False)
            df = df.rename(columns={'title':'Title'})
            df['Title'] = df['Title'].str.upper()
            df_titles['Title'] = df_titles['Title'].str.upper()

            def find_similar_title(title, titles, threshold=80):
                for t in titles:
                    similarity = fuzz.ratio(title, t)
                    if similarity >= threshold:
                        return t
                return None

            # Adding a column to df with the most similar title from df_titles
            df['Similar_Title'] = df['Title'].apply(lambda x: find_similar_title(x, df_titles['Title'], threshold=80))

            # Performing the merge based on the similar titles
            df_not = df.merge(df_titles[['Title']], left_on='Similar_Title', right_on='Title', how='left', indicator=True)
            df_not = df_not[df_not['_merge'] == 'left_only']
            df_not.drop(['_merge', 'Similar_Title'], axis=1, inplace=True)
            df_not = df_not.reset_index(drop=True)
            df_not

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
        event_info = evens_conferences()
        for info in event_info:
            st.write(info)

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