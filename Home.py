from pyzotero import zotero
import pandas as pd
import streamlit as st
from IPython.display import HTML
import streamlit.components.v1 as components
import numpy as np
import altair as alt
from pandas.io.json import json_normalize
from datetime import date, timedelta  
import datetime
from streamlit_extras.switch_page_button import switch_page
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
import gsheetsdb as gdb
import datetime as dt
import time


# Connecting Zotero with API
library_id = '2514686'
library_type = 'group'
api_key = '' # api_key is only needed for private groups and libraries

# Bringing recently changed items

st.set_page_config(layout = "wide", 
                    page_title='Intelligence studies network',
                    page_icon="https://images.pexels.com/photos/315918/pexels-photo-315918.png",
                    initial_sidebar_state="auto") 
pd.set_option('display.max_colwidth', None)

zot = zotero.Zotero(library_id, library_type)

@st.cache_data(ttl=10)
def zotero_data(library_id, library_type):
    items = zot.top(limit=15)

    data=[]
    columns = ['Title','Publication type', 'Link to publication', 'Abstract', 'Zotero link', 'Date added', 'Date published', 'Date modified', 'Col key', 'FirstName', 'Pub_venue']

    for item in items:
        data.append((item['data']['title'], 
        item['data']['itemType'], 
        item['data']['url'], 
        item['data']['abstractNote'], 
        item['links']['alternate']['href'],
        item['data']['dateAdded'],
        item['data'].get('date'), 
        item['data']['dateModified'],
        item['data']['collections'],
        item['data']['creators'],
        item['data'].get('publicationTitle')
        ))
    df = pd.DataFrame(data, columns=columns)
    return df

df = zotero_data(library_id, library_type)

df['Abstract'] = df['Abstract'].replace(r'^\s*$', np.nan, regex=True) # To replace '' with NaN. Otherwise the code below do not understand the value is nan.
df['Abstract'] = df['Abstract'].fillna('No abstract')

split_df= pd.DataFrame(df['Col key'].tolist())
df_fa = df['FirstName']
df_fa = pd.DataFrame(df_fa.tolist())
df_fa = df_fa[0]
df_fa = df_fa.apply(lambda x: {} if pd.isna(x) else x) # https://stackoverflow.com/questions/44050853/pandas-json-normalize-and-null-values-in-json
df_new = pd.json_normalize(df_fa, errors='ignore') 
df = pd.concat([df, split_df, df_new], axis=1)
df['firstName'] = df['firstName'].fillna('null')
df['lastName'] = df['lastName'].fillna('null')

# Change type name
df['Publication type'] = df['Publication type'].replace(['thesis'], 'Thesis')
df['Publication type'] = df['Publication type'].replace(['journalArticle'], 'Journal article')
df['Publication type'] = df['Publication type'].replace(['book'], 'Book')
df['Publication type'] = df['Publication type'].replace(['bookSection'], 'Book chapter')
df['Publication type'] = df['Publication type'].replace(['blogPost'], 'Blog post')
df['Publication type'] = df['Publication type'].replace(['videoRecording'], 'Video')
df['Publication type'] = df['Publication type'].replace(['podcast'], 'Podcast')
df['Publication type'] = df['Publication type'].replace(['magazineArticle'], 'Magazine article')
df['Publication type'] = df['Publication type'].replace(['webpage'], 'Webpage')
df['Publication type'] = df['Publication type'].replace(['newspaperArticle'], 'Newspaper article')
df['Publication type'] = df['Publication type'].replace(['report'], 'Report')
df['Publication type'] = df['Publication type'].replace(['forumPost'], 'Forum post')

df['Date published'] = pd.to_datetime(df['Date published'], errors='coerce')
df['Date published'] = pd.to_datetime(df['Date published'],utc=True).dt.tz_convert('Europe/London')
df['Date published'] = df['Date published'].dt.strftime('%d-%m-%Y')
df['Date published'] = df['Date published'].fillna('No date')
# df['Date published'] = df['Date published'].map(lambda x: x.strftime('%d/%m/%Y') if x else 'No date')

df['Date added'] = pd.to_datetime(df['Date added'], errors='coerce')
df['Date added'] = df['Date added'].dt.strftime('%d/%m/%Y')
df['Date modified'] = pd.to_datetime(df['Date modified'], errors='coerce')
df['Date modified'] = df['Date modified'].dt.strftime('%d/%m/%Y, %H:%M')

# Bringing collections

@st.cache_data(ttl=10)
def zotero_collections2(library_id, library_type):
    bbb = zot.collections()
    data3=[]
    columns3 = ['Key','Name', 'Number', 'Link']
    for item in bbb:
        data3.append((item['data']['key'], item['data']['name'], item['meta']['numItems'], item['links']['alternate']['href']))
    pd.set_option('display.max_colwidth', None)
    df_collections_2 = pd.DataFrame(data3, columns=columns3)
    return df_collections_2
df_collections_2 = zotero_collections2(library_id, library_type)

@st.cache_data
def zotero_collections(library_id, library_type):
    collections = zot.collections()
    data2=[]
    columns2 = ['Key','Name', 'Link']
    for item in collections:
        data2.append((item['data']['key'], item['data']['name'], item['links']['alternate']['href']))
    pd.set_option('display.max_colwidth', None)
    df_collections = pd.DataFrame(data2, columns=columns2)
    return df_collections
df_collections = zotero_collections(library_id, library_type)

df_collections = df_collections.sort_values(by='Name')

# df['Col1Name'] = df['col1'].map(df_collections['Name'])

if 0 in df:
    merged_df = pd.merge(
        left=df,
        right=df_collections,
        left_on=0,
        right_on='Key',
        how='left'
    )
    if 1 in merged_df:
        merged_df = pd.merge(
            left=merged_df,
            right=df_collections,
            left_on=1,
            right_on='Key',
            how='left'
        )
        if 2 in merged_df:
            merged_df = pd.merge(
                left=merged_df,
                right=df_collections,
                left_on=2,
                right_on='Key',
                how='left'
            ) 

df = merged_df.copy()
df = df.fillna('')

# Streamlit app

st.title("Intelligence studies network")
st.header('Intelligence bibliography')
# st.header("[Zotero group library](https://www.zotero.org/groups/2514686/intelligence_bibliography/library)")

into = '''
Wellcome to **Intelligence studies network**! 
This website lists different sources, events, conferences, and call for papers on intelligence history and intelligence studies. 
The current page shows the recently added or updated items. 
**If you wish to see more sources under different themes, see the sidebar menu** :arrow_left: .
The website has also a dynamic [digest](https://intelligence.streamlit.app/Digest) that you can tract latest publications & events.
Check it out the [short guide](https://medium.com/@yaliozkan/introduction-to-intelligence-studies-network-ed63461d1353) for a quick intoduction.'''


with st.spinner('Retrieving data & updating dashboard...'):

    count = zot.count_items()

    col1, col2 = st.columns([3,5])
    with col2:
        with st.expander('Intro'):
            st.info(into)
    with col1:
        st.write('There are '+  '**'+str(count)+ '**' + ' items in the [Intelligence bibliography Zotero group library](https://www.zotero.org/groups/2514686/intelligence_bibliography/items).')
        st.write('The library last updated on ' + '**'+ df.loc[0]['Date modified']+'**')

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
        with st.expander('Disclaimer'):
            st.warning('''
            This website and the Intelligence bibliography Zotero group library do not list all the sources on intelligence studies. 
            The list is created based on the creator's subjective views.
            ''')
        with st.expander('Contact us'):
            st.write('If you have any questions or suggestions, please do get in touch with us by filling the form [here](https://www.intelligencenetwork.org/contact-us).')
            st.write('Report your technical issues or requests [here](https://github.com/YusufAliOzkan/zotero-intelligence-bibliography/issues).')
        st.write('See our dynamic [digest](https://intelligence.streamlit.app/Digest)')
    # Recently added items

    tab1, tab2 = st.tabs(['📑 Publications', '📊 Dashboard'])
    with tab1:
        col1, col2 = st.columns([5,2]) 
        with col1:
            st.header('Recently added or updated items: ')
                
            df_download = df.iloc[:, [0,1,2,4,5,6,14,15]] 
            df_download['First author'] = df['firstName'] + ' ' + df['lastName']
            df_download = df_download[['Title', 'Publication type', 'First author', 'Link to publication', 'Zotero link', 'Date published', 'Date added']]

            def convert_df(df):
                return df.to_csv(index=False).encode('utf-8-sig') # not utf-8 because of the weird character,  Â cp1252
            csv = convert_df(df_download)
            # csv = df_download
            # # st.caption(collection_name)
            today = datetime.date.today().isoformat()
            a = 'intelligence-bibliography-' + today
            st.download_button('💾 Download recently added items', csv, (a+'.csv'), mime="text/csv", key='download-csv')

            with st.expander('Click to hide the list', expanded=True):
                display = st.checkbox('Display theme and abstract')

                df_last = ('**'+ df['Publication type']+ '**'+ ': ' + df['Title'] +', ' +                        
                            ' (by ' + '*' + df['firstName'] + '*'+ ' ' + '*' + df['lastName'] + '*' + ') ' +
                            ' (Published on: ' + df['Date published']+', ' +
                            'Added on: ' + df['Date added']+')'+
                            '[[Publication link]]'+ '('+ df['Link to publication'] + ')' +
                            "[[Zotero link]]" +'('+ df['Zotero link'] + ')' 
                            )

                row_nu_1 = len(df_last.index)
                for i in range(row_nu_1):
                    publication_type = df['Publication type'].iloc[i]
                    if publication_type in ["Journal article", "Magazine article", 'Newspaper article']:
                        df_last = ('**'+ df['Publication type']+ '**'+ ': ' + df['Title'] +', ' +                        
                                    ' (by ' + '*' + df['firstName'] + '*'+ ' ' + '*' + df['lastName'] + '*' + ') ' +
                                    ' (Published on: ' + df['Date published']+') ' +
                                    " (Published in: " + "*" + df['Pub_venue'] + "*" + ') '+
                                    '[[Publication link]]'+ '('+ df['Link to publication'] + ')' +
                                    "[[Zotero link]]" +'('+ df['Zotero link'] + ')' 
                                    )
                        st.write(f"{i+1}) " + df_last.iloc[i])
                    else:
                        df_last = ('**'+ df['Publication type']+ '**'+ ': ' + df['Title'] +', ' +                        
                                    ' (by ' + '*' + df['firstName'] + '*'+ ' ' + '*' + df['lastName'] + '*' + ') ' +
                                    ' (Published on: ' + df['Date published']+', ' +
                                    'Added on: ' + df['Date added']+') '+
                                    '[[Publication link]]'+ '('+ df['Link to publication'] + ')' +
                                    "[[Zotero link]]" +'('+ df['Zotero link'] + ')' 
                                    )
                        st.write(f"{i+1}) " + df_last.iloc[i])
                    if display:
                        a=''
                        b=''
                        c=''
                        if 'Name_x' in df:
                            a= '['+'['+df['Name_x'].iloc[i]+']' +'('+ df['Link_x'].iloc[i] + ')'+ ']'
                            if 'Name_y' in df:
                                b='['+'['+df['Name_y'].iloc[i]+']' +'('+ df['Link_y'].iloc[i] + ')' +']'
                                if df['Name_y'].iloc[i]=='':
                                    b=''
                                if 'Name' in df:
                                    c= '['+'['+df['Name'].iloc[i]+']' +'('+ df['Link'].iloc[i] + ')'+ ']'
                                    if df['Name'].iloc[i]=='':
                                        c=''
                        else:
                            st.caption('No theme to display!')
                        st.caption('Theme(s):  \n ' + a + ' ' +b+ ' ' + c)
                        st.caption('Abstract: '+ df['Abstract'].iloc[i])
        

        # Items by Collection list

            # st.header('Items by collection: ')
            # clist = df_collections['Name'].unique()
            # collection_name = st.selectbox('Select a collection:', clist)
            # collection_code = df_collections.loc[df_collections['Name']==collection_name, 'Key'].values[0]

            # df_collections=df_collections['Name'].reset_index()
            # pd.set_option('display.max_colwidth', None)

            # # Collection items

            # count_collection = zot.num_collectionitems(collection_code)

            # items = zot.everything(zot.collection_items_top(collection_code))

            # data3=[]
            # columns3=['Title','Publication type', 'Link to publication', 'Abstract', 'Zotero link', 'Date published', 'FirstName2']

            # for item in items:
            #     data3.append((
            #         item['data']['title'], 
            #         item['data']['itemType'], 
            #         item['data']['url'], 
            #         item['data']['abstractNote'], 
            #         item['links']['alternate']['href'],
            #         item['data'].get('date'),
            #         item['data']['creators']
            #         )) 
            # pd.set_option('display.max_colwidth', None)

            # df = pd.DataFrame(data3, columns=columns3)
            
            # df['Publication type'] = df['Publication type'].replace(['thesis'], 'Thesis')
            # df['Publication type'] = df['Publication type'].replace(['journalArticle'], 'Journal article')
            # df['Publication type'] = df['Publication type'].replace(['book'], 'Book')
            # df['Publication type'] = df['Publication type'].replace(['bookSection'], 'Book chapter')
            # df['Publication type'] = df['Publication type'].replace(['blogPost'], 'Blog post')
            # df['Publication type'] = df['Publication type'].replace(['videoRecording'], 'Video')
            # df['Publication type'] = df['Publication type'].replace(['podcast'], 'Podcast')
            # df['Publication type'] = df['Publication type'].replace(['magazineArticle'], 'Magazine article')
            # df['Publication type'] = df['Publication type'].replace(['webpage'], 'Webpage')
            # df['Publication type'] = df['Publication type'].replace(['newspaperArticle'], 'Newspaper article')
            # df['Publication type'] = df['Publication type'].replace(['report'], 'Report')
            # df['Publication type'] = df['Publication type'].replace(['forumPost'], 'Forum post')

            # df['Date published'] = pd.to_datetime(df['Date published'],utc=True).dt.tz_convert('Europe/London')
            # df['Date published'] = df['Date published'].dt.strftime('%d-%m-%Y')
            # df['Date published'] = df['Date published'].fillna('No date')    

            # if df['FirstName2'].any() in ("", [], None, 0, False):
            #     # st.write('no author')
            #     df['firstName'] = 'null'
            #     df['lastName'] = 'null'

            #     df_items = ('**'+ df['Publication type']+ '**'+ ': ' +
            #         df['Title'] + ' '+ 
            #         ' (by ' + '*' + df['firstName'] + '*'+ ' ' + '*' + df['lastName'] + '*' + ') ' + 
            #         "[[Publication link]]" +'('+ df['Link to publication'] + ')' +'  '+
            #         "[[Zotero link]]" +'('+ df['Zotero link'] + ')' +
            #         ' (Published on: ' +df['Date published'] + ')'
            #         )
            # else:
            #     # st.write('author entered')
            #     ## This section is for displaying the first author details but it doesn't work for now because of json normalization error.
            #     df_fa = df['FirstName2']
            #     df_fa = pd.DataFrame(df_fa.tolist())
            #     df_fa = df_fa[0]
            #     df_fa = df_fa.apply(lambda x: {} if pd.isna(x) else x) # https://stackoverflow.com/questions/44050853/pandas-json-normalize-and-null-values-in-json
            #     df_new = pd.json_normalize(df_fa, errors='ignore') 
            #     df = pd.concat([df, df_new], axis=1)
            #     df['firstName'] = df['firstName'].fillna('null')
            #     df['lastName'] = df['lastName'].fillna('null')
                
            #     df_items = ('**'+ df['Publication type']+ '**'+ ': ' +
            #                 df['Title'] + ' '+ 
            #                 ' (by ' + '*' + df['firstName'] + '*'+ ' ' + '*' + df['lastName'] + '*' + ') ' + # IT CANNOT READ THE NAN VALUES
            #                 "[[Publication link]]" +'('+ df['Link to publication'] + ')' +'  '+
            #                 "[[Zotero link]]" +'('+ df['Zotero link'] + ')' +
            #                 ' (Published on: ' +df['Date published'] + ')'
            #                 )

            # row_nu_1= len(df.index)
            # if row_nu_1<15:
            #     row_nu_1=row_nu_1
            # else:
            #     row_nu_1=15

            # st.markdown('#### Collection theme: ' + collection_name)
            # st.caption('This collection has ' + str(count_collection) + ' items.')
            # with st.expander("Expand to see the list", expanded=False):
            #     st.write('This list shows the last 15 added items. To see the full collection list click [here](https://www.zotero.org/groups/2514686/intelligence_bibliography/collections/' + collection_code + ')')
            #     # display2 = st.checkbox('Display abstracts')
            #     for i in range(row_nu_1):
            #         st.write(''+str(i+1)+') ' +df_items.iloc[i])
            #         df_items.fillna("nan") 
            #         # if display2:
            #         #     st.caption(df['Abstract'].iloc[i])

        with col2:
            with st.expander("Collections in Zotero library", expanded=False):
                row_nu_collections = len(df_collections_2.index)        
                for i in range(row_nu_collections):
                    st.caption('[' + df_collections_2.sort_values(by='Name')['Name'].iloc[i]+ ']'+ '('+ df_collections_2.sort_values(by='Name')['Link'].iloc[i] + ')' + 
                    ' [' + str(df_collections_2.sort_values(by='Name')['Number'].iloc[i]) + ' items]'
                    )
            with st.expander('Collections in this site', expanded=False):
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
                st.caption('[Special collections](https://intelligence.streamlit.app/Special_collections)')

            with st.expander('Events & conferences', expanded=True):
                st.markdown('##### Next event')
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

    with tab2:
        st.header('Dashboard')
        number0 = st.slider('Select a number collections', 3,30,15)
        df_collections_2.set_index('Name', inplace=True)
        df_collections_2 = df_collections_2.sort_values(['Number'], ascending=[False])
        plot= df_collections_2.head(number0+1)
        # st.bar_chart(plot['Number'].sort_values(), height=600, width=600, use_container_width=True)
        plot = plot.reset_index()

        plot = plot[plot['Name']!='01 Intelligence history']
        fig = px.bar(plot, x='Name', y='Number', color='Name')
        fig.update_layout(
            autosize=False,
            width=600,
            height=600,)
        fig.update_layout(title={'text':'Top ' + str(number0) + ' collections in the library', 'y':0.95, 'x':0.4, 'yanchor':'top'})
        st.plotly_chart(fig, use_container_width = True)

        # Visauls for all items in the library
        df_csv = pd.read_csv('all_items.csv')
        df_csv['Date published'] = pd.to_datetime(df_csv['Date published'],utc=True, errors='coerce').dt.tz_convert('Europe/London')
        df_csv['Date year'] = df_csv['Date published'].dt.strftime('%Y')
        df_csv['Date year'] = df_csv['Date year'].fillna('No date')
        df = df_csv.copy()
        df_year=df_csv['Date year'].value_counts()
        df_year=df_year.reset_index()
        df_year=df_year.rename(columns={'index':'Publication year','Date year':'Count'})
        df_year.drop(df_year[df_year['Publication year']== 'No date'].index, inplace = True)
        df_year=df_year.sort_values(by='Publication year', ascending=True)
        df_year=df_year.reset_index(drop=True)
        max_y = int(df_year['Publication year'].max())
        min_y = int(df_year['Publication year'].min())

        with st.expander('Select parameters', expanded=False):
            types = st.multiselect('Publication type', df_csv['Publication type'].unique(),df_csv['Publication type'].unique())        
            years = st.slider('Publication years between:', min_y, max_y, (min_y,max_y), key='years')
            if st.button('Update dashboard'):
                df_csv = df_csv[df_csv['Publication type'].isin(types)]
                df_csv = df_csv[df_csv['Date year'] !='No date']
                filter = (df_csv['Date year'].astype(int)>=years[0]) & (df_csv['Date year'].astype(int)<years[1])
                df_csv = df_csv.loc[filter]
                df_year=df_csv['Date year'].value_counts()
                df_year=df_year.reset_index()
                df_year=df_year.rename(columns={'index':'Publication year','Date year':'Count'})
                df_year.drop(df_year[df_year['Publication year']== 'No date'].index, inplace = True)
                df_year=df_year.sort_values(by='Publication year', ascending=True)
                df_year=df_year.reset_index(drop=True)

        df_types = pd.DataFrame(df_csv['Publication type'].value_counts())
        df_types = df_types.sort_values(['Publication type'], ascending=[False])
        df_types=df_types.reset_index()
        df_types = df_types.rename(columns={'index':'Publication type','Publication type':'Count'})

        if df_csv['Title'].any() in ("", [], None, 0, False):
            st.write('No data to visualise')
            st.stop()
        col1, col2 = st.columns(2)
        with col1:

            log0 = st.checkbox('Show in log scale', key='log0')

            if log0:
                fig = px.bar(df_types, x='Publication type', y='Count', color='Publication type', log_y=True)
                fig.update_layout(
                    autosize=False,
                    width=1200,
                    height=600,)
                fig.update_xaxes(tickangle=-70)
                fig.update_layout(title={'text':'Item types in log scale', 'y':0.95, 'x':0.4, 'yanchor':'top'})
                col1.plotly_chart(fig, use_container_width = True)
            else:
                fig = px.bar(df_types, x='Publication type', y='Count', color='Publication type')
                fig.update_layout(
                    autosize=False,
                    width=1200,
                    height=600,)
                fig.update_xaxes(tickangle=-70)
                fig.update_layout(title={'text':'Item types', 'y':0.95, 'x':0.4, 'yanchor':'top'})
                col1.plotly_chart(fig, use_container_width = True)

        with col2:
            fig = px.pie(df_types, values='Count', names='Publication type')
            fig.update_layout(title={'text':'Item types', 'y':0.95, 'x':0.45, 'yanchor':'top'})
            col2.plotly_chart(fig, use_container_width = True)

        col1, col2 = st.columns(2)
        with col1:
            fig = px.bar(df_year, x='Publication year', y='Count')
            fig.update_xaxes(tickangle=-70)
            fig.update_layout(
                autosize=False,
                width=1200,
                height=600,)
            fig.update_layout(title={'text':'All items in the library by publication year', 'y':0.95, 'x':0.5, 'yanchor':'top'})
            col1.plotly_chart(fig, use_container_width = True)

        with col2:
            df_year['Sum'] = df_year['Count'].cumsum()
            fig2 = px.line(df_year, x='Publication year', y='Sum')
            fig2.update_layout(title={'text':'All items in the library by publication year (cumulative sum)', 'y':0.95, 'x':0.5, 'yanchor':'top'})
            fig2.update_layout(
                autosize=False,
                width=1200,
                height=600,)
            fig2.update_xaxes(tickangle=-70)
            col2.plotly_chart(fig2, use_container_width = True)

        col1, col2 = st.columns(2)
        with col1:
            number = st.slider('Select a number of publishers', 0, 30, 10)
            df_publisher = pd.DataFrame(df_csv['Publisher'].value_counts())
            df_publisher = df_publisher.sort_values(['Publisher'], ascending=[False])
            df_publisher = df_publisher.reset_index()
            df_publisher = df_publisher.rename(columns={'index':'Publisher','Publisher':'Count'})
            df_publisher = df_publisher.head(number)

            log1 = st.checkbox('Show in log scale', key='log1')
            leg1 = st.checkbox('Disable legend', key='leg1', disabled=False)

            if df_publisher['Publisher'].any() in ("", [], None, 0, False):
                st.write('No publisher to display')
            else:
                if log1:
                    if leg1:
                        fig = px.bar(df_publisher, x='Publisher', y='Count', color='Publisher', log_y=True)
                        fig.update_layout(
                            autosize=False,
                            width=1200,
                            height=700,
                            showlegend=False)
                        fig.update_xaxes(tickangle=-70)
                        fig.update_layout(title={'text':'Top ' + str(number) + ' publishers (in log scale)', 'y':0.95, 'x':0.4, 'yanchor':'top'})
                        col1.plotly_chart(fig, use_container_width = True)
                    else:
                        fig = px.bar(df_publisher, x='Publisher', y='Count', color='Publisher', log_y=True)
                        fig.update_layout(
                            autosize=False,
                            width=1200,
                            height=700,
                            showlegend=True)
                        fig.update_xaxes(tickangle=-70)
                        fig.update_layout(title={'text':'Top ' + str(number) + ' publishers (in log scale)', 'y':0.95, 'x':0.4, 'yanchor':'top'})
                        col1.plotly_chart(fig, use_container_width = True)
                else:
                    if leg1:
                        fig = px.bar(df_publisher, x='Publisher', y='Count', color='Publisher', log_y=False)
                        fig.update_layout(
                            autosize=False,
                            width=1200,
                            height=700,
                            showlegend=False)
                        fig.update_xaxes(tickangle=-70)
                        fig.update_layout(title={'text':'Top ' + str(number) + ' publishers', 'y':0.95, 'x':0.4, 'yanchor':'top'})
                        col1.plotly_chart(fig, use_container_width = True)
                    else:
                        fig = px.bar(df_publisher, x='Publisher', y='Count', color='Publisher', log_y=False)
                        fig.update_layout(
                            autosize=False,
                            width=1200,
                            height=700,
                            showlegend=True)
                        fig.update_xaxes(tickangle=-70)
                        fig.update_layout(title={'text':'Top ' + str(number) + ' publishers', 'y':0.95, 'x':0.4, 'yanchor':'top'})
                        col1.plotly_chart(fig, use_container_width = True)
                with st.expander('See publishers'):
                    row_nu_collections = len(df_publisher.index)        
                    for i in range(row_nu_collections):
                        st.caption(df_publisher['Publisher'].iloc[i]
                        )

        with col2:
            number2 = st.slider('Select a number of journals', 0,30,10)
            df_journal = df_csv.loc[df_csv['Publication type']=='Journal article']
            df_journal = pd.DataFrame(df_journal['Journal'].value_counts())
            df_journal = df_journal.sort_values(['Journal'], ascending=[False])
            df_journal = df_journal.reset_index()
            df_journal = df_journal.rename(columns={'index':'Journal','Journal':'Count'})
            df_journal = df_journal.head(number2)

            log2 = st.checkbox('Show in log scale', key='log2')
            leg2 = st.checkbox('Disable legend', key='leg2')

            if df_journal['Journal'].any() in ("", [], None, 0, False):
                st.write('No journal to display')
            else:
                if log2:
                    if leg2:
                        fig = px.bar(df_journal, x='Journal', y='Count', color='Journal', log_y=True)
                        fig.update_layout(
                            autosize=False,
                            width=1200,
                            height=700,
                            showlegend=False)
                        fig.update_xaxes(tickangle=-70)
                        fig.update_layout(title={'text':'Top ' + str(number2) + ' journals that publish intelligence articles (in log scale)', 'y':0.95, 'x':0.4, 'yanchor':'top'})
                        col2.plotly_chart(fig, use_container_width = True)
                    else:
                        fig = px.bar(df_journal, x='Journal', y='Count', color='Journal', log_y=True)
                        fig.update_layout(
                            autosize=False,
                            width=1200,
                            height=700,
                            showlegend=True)
                        fig.update_xaxes(tickangle=-70)
                        fig.update_layout(title={'text':'Top ' + str(number2) + ' journals that publish intelligence articles (in log scale)', 'y':0.95, 'x':0.4, 'yanchor':'top'})
                        col2.plotly_chart(fig, use_container_width = True)
                else:
                    if leg2:
                        fig = px.bar(df_journal, x='Journal', y='Count', color='Journal', log_y=False)
                        fig.update_layout(
                            autosize=False,
                            width=1200,
                            height=700,
                            showlegend=False)
                        fig.update_xaxes(tickangle=-70)
                        fig.update_layout(title={'text':'Top ' + str(number2) + ' journals that publish intelligence articles', 'y':0.95, 'x':0.4, 'yanchor':'top'})
                        col2.plotly_chart(fig, use_container_width = True)
                    else:
                        fig = px.bar(df_journal, x='Journal', y='Count', color='Journal', log_y=False)
                        fig.update_layout(
                            autosize=False,
                            width=1200,
                            height=700,
                            showlegend=True)
                        fig.update_xaxes(tickangle=-70)
                        fig.update_layout(title={'text':'Top ' + str(number2) + ' journals that publish intelligence articles', 'y':0.95, 'x':0.4, 'yanchor':'top'})
                        col2.plotly_chart(fig, use_container_width = True)
                with st.expander('See journals'):
                    row_nu_collections = len(df_journal.index)        
                    for i in range(row_nu_collections):
                        st.caption(df_journal['Journal'].iloc[i]
                        )

        st.write('---')
        df=df_csv.copy()
        def clean_text (text):
            text = text.lower() # lowercasing
            text = re.sub(r'[^\w\s]', ' ', text) # this removes punctuation
            text = re.sub('[0-9_]', ' ', text) # this removes numbers
            text = re.sub('[^a-z_]', ' ', text) # removing all characters except lowercase letters
            return text
        df['clean_title'] = df['Title'].apply(clean_text)
        df['clean_abstract'] = df['Abstract'].astype(str).apply(clean_text)
        df_abs_no = df.dropna(subset=['clean_abstract'])
        df['clean_title'] = df['clean_title'].apply(lambda x: ' '.join ([w for w in x.split() if len (w)>2])) # this function removes words less than 2 words
        df['clean_abstract'] = df['clean_abstract'].apply(lambda x: ' '.join ([w for w in x.split() if len (w)>2])) # this function removes words less than 2 words

        def tokenization(text):
            text = re.split('\W+', text)
            return text
        df['token_title']=df['clean_title'].apply(tokenization)
        df['token_abstract']=df['clean_abstract'].apply(tokenization)
        stopword = nltk.corpus.stopwords.words('english')

        SW = ['york', 'intelligence', 'security', 'pp', 'war','world', 'article', 'twitter', 'nan',
            'new', 'isbn', 'book', 'also', 'yet', 'matter', 'erratum', 'commentary', 'studies',
            'volume', 'paper', 'study', 'question', 'editorial', 'welcome', 'introduction', 'editorial', 'reader',
            'university', 'followed', 'particular', 'based', 'press', 'examine', 'show', 'may', 'result', 'explore',
            'examines', 'become', 'used', 'journal', 'london', 'review']
        stopword.extend(SW)

        def remove_stopwords(text):
            text = [i for i in text if i] # this part deals with getting rid of spaces as it treads as a string
            text = [word for word in text if word not in stopword] #keep the word if it is not in stopword
            return text
        df['stopword']=df['token_title'].apply(remove_stopwords)
        df['stopword_abstract']=df['token_abstract'].apply(remove_stopwords)

        wn = nltk.WordNetLemmatizer()
        def lemmatizer(text):
            text = [wn.lemmatize(word) for word in text]
            return text

        df['lemma_title'] = df['stopword'].apply(lemmatizer) # error occurs in this line
        df['lemma_abstract'] = df['stopword_abstract'].apply(lemmatizer) # error occurs in this line

        listdf = df['lemma_title']
        listdf_abstract = df['lemma_abstract']

        st.markdown('## Wordcloud')
        wordcloud_opt = st.radio('Wordcloud of:', ('Titles', 'Abstracts'))
        if wordcloud_opt=='Titles':
            df_list = [item for sublist in listdf for item in sublist]
            string = pd.Series(df_list).str.cat(sep=' ')
            wordcloud_texts = string
            wordcloud_texts_str = str(wordcloud_texts)
            wordcloud = WordCloud(stopwords=stopword, width=1500, height=750, background_color='white', collocations=False, colormap='magma').generate(wordcloud_texts_str)
            plt.figure(figsize=(20,8))
            plt.axis('off')
            plt.title('Top words in title (Intelligence bibliography collection)')
            plt.imshow(wordcloud)
            plt.axis("off")
            plt.show()
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.pyplot() 
        else:
            st.warning('Please bear in mind that not all items listed in this bibliography have an abstract. Therefore, this wordcloud should not be considered as authoritative. The number of items that have an abstract is ' + str(len(df_abs_no))+'.')
            df_list_abstract = [item for sublist in listdf_abstract for item in sublist]
            string = pd.Series(df_list_abstract).str.cat(sep=' ')
            wordcloud_texts = string
            wordcloud_texts_str = str(wordcloud_texts)
            wordcloud = WordCloud(stopwords=stopword, width=1500, height=750, background_color='white', collocations=False, colormap='magma').generate(wordcloud_texts_str)
            plt.figure(figsize=(20,8))
            plt.axis('off')
            plt.title('Top words in abstract (Intelligence bibliography collection)')
            plt.imshow(wordcloud)
            plt.axis("off")
            plt.show()
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.pyplot() 

        ## Bring everything in the library
        # types = zot.everything(zot.top())

        # data_t=[]
        # columns_t = ['Publication type']

        # for item in types:
        #     data_t.append((item['data']['itemType']))

        # pd.set_option('display.max_colwidth', None)
        # df_t = pd.DataFrame(data_t, columns=columns_t)

        # df_t['Publication type'] = df_t['Publication type'].replace(['thesis'], 'Thesis')
        # df_t['Publication type'] = df_t['Publication type'].replace(['journalArticle'], 'Journal article')
        # df_t['Publication type'] = df_t['Publication type'].replace(['book'], 'Book')
        # df_t['Publication type'] = df_t['Publication type'].replace(['bookSection'], 'Book chapter')
        # df_t['Publication type'] = df_t['Publication type'].replace(['blogPost'], 'Blog post')
        # df_t['Publication type'] = df_t['Publication type'].replace(['videoRecording'], 'Video')
        # df_t['Publication type'] = df_t['Publication type'].replace(['podcast'], 'Podcast')
        # df_t['Publication type'] = df_t['Publication type'].replace(['magazineArticle'], 'Magazine article')
        # df_t['Publication type'] = df_t['Publication type'].replace(['webpage'], 'Webpage')
        # df_t['Publication type'] = df_t['Publication type'].replace(['newspaperArticle'], 'Newspaper article')
        # df_t['Publication type'] = df_t['Publication type'].replace(['report'], 'Report')


        # df_types = pd.DataFrame(df_t['Publication type'].value_counts())

        # st.header('Items in the library by type: ')
        # df_types = df_types.sort_values(['Publication type'], ascending=[False])
        # plot2= df_types.head(10)

        # st.bar_chart(plot2['Publication type'].sort_values(), height=600, width=600, use_container_width=True)



    st.write('---')
    with st.expander('Acknowledgements'):
        st.subheader('Acknowledgements')
        st.write('The following sources are used to collate some of the items and events in this website:')
        st.write("1. [King's Intelligence and Security Group (KISG) digest](https://kisg.co.uk/kisg-digests) compiled by David Schaefer")
        st.write("2. [International Association for Intelligence Education (IAIE) digest](https://www.iafie.org/Login.aspx) compiled by Filip Kovacevic")
        st.write("3. [North American Society for Intelligence History (NASIH)](https://www.intelligencehistory.org/brownbags)")

    components.html(
    """
    <a rel="license" href="http://creativecommons.org/licenses/by/4.0/"><img alt="Creative Commons Licence" style="border-width:0" 
    src="https://i.creativecommons.org/l/by/4.0/80x15.png" /></a><br />
    © 2022 All rights reserved. This website is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">Creative Commons Attribution 4.0 International License</a>.
    """
    )    