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
                    page_title='Intelligence studies network',
                    page_icon="https://images.pexels.com/photos/315918/pexels-photo-315918.png",
                    initial_sidebar_state="auto") 

st.title("Special collections")

with st.spinner('Retrieving data & updating dashboard...'):

    # Connecting Zotero with API
    library_id = '2514686' # intel 2514686
    library_type = 'group'
    api_key = '' # api_key is only needed for private groups and libraries


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

    zot = zotero.Zotero(library_id, library_type)

    @st.cache_data(ttl=300)
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
    df_collections=df_collections[df_collections['Name'].str.contains("98.")]
    df_collections = df_collections.iloc[2: , :]

    st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
    container = st.container()

    # clist = df_collections['Name'].unique()

    tab1, tab2, tab3 = st.tabs(['📑 Publications', '📊 Dashboard', 'Source suggestion'])
    with tab1:
        col1, col2 = st.columns([5,1.6])
        with col1:
            radio = container.radio('Select a collection', df_collections['Name'])
            # collection_name = st.selectbox('Select a collection:', clist)
            collection_name = radio
            collection_code = df_collections.loc[df_collections['Name']==collection_name, 'Key'].values[0]

            df_collections=df_collections['Name'].reset_index()
            pd.set_option('display.max_colwidth', None)

            count_collection = zot.num_collectionitems(collection_code)

            @st.cache_data(ttl=300)
            def get_zotero_data(collection_code):
                items = zot.everything(zot.collection_items_top(collection_code))
                data3=[]
                columns3=['Title','Publication type', 'Link to publication', 'Abstract', 'Zotero link', 'Date published', 'FirstName2', 'Publisher', 'Journal']
                for item in items:
                    data3.append((
                        item['data']['title'], 
                        item['data']['itemType'], 
                        item['data']['url'], 
                        item['data']['abstractNote'], 
                        item['links']['alternate']['href'],
                        item['data'].get('date'),
                        item['data']['creators'],
                        item['data'].get('publisher'),
                        item['data'].get('publicationTitle')
                        )) 
                pd.set_option('display.max_colwidth', None)
                df = pd.DataFrame(data3, columns=columns3)
                return df
            df = get_zotero_data(collection_code)

            # df['Date published'] = pd.to_datetime(df['Date published'], errors='coerce')
            # df['Date published'] = df['Date published'].map(lambda x: x.strftime('%d/%m/%Y') if x else 'No date')
            # df

            # Change type name
            type_map = {
                'thesis': 'Thesis',
                'journalArticle': 'Journal article',
                'book': 'Book',
                'bookSection': 'Book chapter',
                'blogPost': 'Blog post',
                'videoRecording': 'Video',
                'podcast': 'Podcast',
                'magazineArticle': 'Magazine article',
                'webpage': 'Webpage',
                'newspaperArticle': 'Newspaper article',
                'report': 'Report',
                'forumPost': 'Forum post',
                'conferencePaper' : 'Conference paper',
                'presentation' : 'Presentation',
                'audioRecording':'Podcast'
            }
            df['Publication type'] = df['Publication type'].replace(type_map)

            df['Publisher'] = df['Publisher'].replace(['Taylor & Francis Group', 'Taylor and Francis', 'Taylor & Francis'], 'Taylor and Francis')
            df['Publisher'] = df['Publisher'].replace(['Routledge', 'Routledge Handbooks Online'], 'Routledge')
            df['Publisher'] = df['Publisher'].replace(['Praeger Security International', 'Praeger'], 'Praeger')

            df['Journal'] = df['Journal'].replace(['International Journal of Intelligence and Counter-Intelligence', 'International Journal of Intelligence and Counterintelligence', 'International Journal of Intelligence and CounterIntelligence'], 'Intl Journal of Intelligence and Counterintelligence')
            df['Journal'] = df['Journal'].replace(['Intelligence and national security', 'Intelligence and National Security', 'Intelligence & National Security'], 'Intelligence and National Security')

            df['Date published'] = pd.to_datetime(df['Date published'],utc=True, errors='coerce').dt.tz_convert('Europe/London')
            df['Date published'] = df['Date published'].dt.strftime('%Y-%m-%d')
            df['Date published'] = df['Date published'].fillna('No date')
            
            # sort = st.checkbox('Sort by publication date', disabled=False)
            # if sort:
            #     df['Date published'] = df['Date published'].astype('datetime64[ns]')
            #     df=df.sort_values(by='Date published', ascending=True)

            st.markdown('#### Collection theme: ' + collection_name)
            st.caption('This collection has ' + str(count_collection) + ' items (this number may include reviews attached to sources).') # count_collection

            df2 = df.copy()
            types = st.multiselect('Publication type', df['Publication type'].unique(),df['Publication type'].unique())
            df = df[df['Publication type'].isin(types)]  #filtered_df = df[df["app"].isin(selected_options)]
            df = df.reset_index()

            if df['FirstName2'].any() in ("", [], None, 0, False):
                # st.write('no author')
                df['firstName'] = 'null'
                df['lastName'] = 'null'

                df_items = ('**'+ df['Publication type']+ '**'+ ': ' +
                    df['Title'] + ' '+ 
                    ' (by ' + '*' + df['firstName'] + '*'+ ' ' + '*' + df['lastName'] + '*' + ') ' + 
                    "[[Publication link]]" +'('+ df['Link to publication'] + ')' +'  '+
                    "[[Zotero link]]" +'('+ df['Zotero link'] + ')' +
                    ' (Published on: ' +df['Date published'] + ')'
                    )
            else:
                # st.write('author entered')
                ## This section is for displaying the first author details but it doesn't work for now because of json normalization error.
                df_fa = df['FirstName2']
                df_fa = pd.DataFrame(df_fa.tolist())
                df_fa = df_fa[0]
                df_fa = df_fa.apply(lambda x: {} if pd.isna(x) else x) # https://stackoverflow.com/questions/44050853/pandas-json-normalize-and-null-values-in-json
                df_new = pd.json_normalize(df_fa, errors='ignore') 
                df = pd.concat([df, df_new], axis=1)
                df['firstName'] = df['firstName'].fillna('null')
                df['lastName'] = df['lastName'].fillna('null')
                
                df_items = ('**'+ df['Publication type']+ '**'+ ': ' +
                            df['Title'] + ' '+ 
                            ' (by ' + '*' + df['firstName'] + '*'+ ' ' + '*' + df['lastName'] + '*' + ') ' + # IT CANNOT READ THE NAN VALUES
                            "[[Publication link]]" +'('+ df['Link to publication'] + ')' +'  '+
                            "[[Zotero link]]" +'('+ df['Zotero link'] + ')' +
                            ' (Published on: ' +df['Date published'] + ')'
                            )

            row_nu_1= len(df.index)
            # if row_nu_1<15:
            #     row_nu_1=row_nu_1
            # else:
            #     row_nu_1=15

            df['First author'] = df['firstName'] + ' ' + df['lastName']
            df_download = df[['Title', 'Publication type', 'First author', 'Link to publication', 'Zotero link', 'Date published']]

            def convert_df(df):
                return df.to_csv(index=False).encode('utf-8-sig') # not utf-8 because of the weird character,  Â cp1252
            today = datetime.date.today().isoformat()
            csv = convert_df(df_download)
            # csv = df_download
            # # st.caption(collection_name)
            st.download_button('💾 Download the collection', csv, collection_name+ '-'+today +'.csv', mime="text/csv", key='download-csv')

            with st.expander("Expand to see the list", expanded=True):
                st.write('To see the collection in Zotero click [here](https://www.zotero.org/groups/2514686/intelligence_bibliography/collections/' + collection_code + ')')

                sort_by_type = st.checkbox('Sort by publication type', key='type')
                display2 = st.checkbox('Display abstracts')

                if sort_by_type:
                    df3=df.copy()
                    df = df.sort_values(by=['Publication type'], ascending=True)
                    types = df['Publication type'].unique()
                    types = pd.DataFrame(types, columns=['Publication type'])
                    row_nu_types = len(types.index)
                    for i in range(row_nu_types):
                        st.subheader(types['Publication type'].iloc[i])
                        b = types['Publication type'].iloc[i]
                        df_a = df[df['Publication type']==b]
                        df_items = ('**'+ df_a['Publication type']+ '**'+ ': ' +
                            df_a['Title'] + ' '+ 
                            ' (by ' + '*' + df_a['firstName'] + '*'+ ' ' + '*' + df_a['lastName'] + '*' + ') ' +
                            ' (Published on: ' +df_a['Date published'] + ') '+
                            "[[Publication link]]" +'('+ df_a['Link to publication'] + ')' +'  '+
                            "[[Zotero link]]" +'('+ df_a['Zotero link'] + ')'
                            )
                        row_nu_1 = len(df_a.index)
                        for i in range(row_nu_1):
                            if df_a['Publication type'].iloc[i] in ['Journal article', 'Magazine article', 'Newspaper article']:
                                df_items = ('**'+ df_a['Publication type']+ '**'+ ': ' +
                                    df_a['Title'] + ' '+ 
                                    ' (by ' + '*' + df_a['firstName'] + '*'+ ' ' + '*' + df_a['lastName'] + '*' + ') ' +
                                    ' (Published on: ' +df_a['Date published'] + ') '+
                                    ' (Published in: ' + '*' + df_a['Journal'].iloc[i] + '*' + ') ' +
                                    "[[Publication link]]" +'('+ df_a['Link to publication'] + ')' +'  '+
                                    "[[Zotero link]]" +'('+ df_a['Zotero link'] + ')'
                                    ) 
                                st.write('' + str(i+1) + ') ' + df_items.iloc[i])
                            else:
                                st.write('' + str(i+1) + ') ' + df_items.iloc[i])
                            df_items.fillna("nan") 
                            if display2:
                                st.caption(df['Abstract'].iloc[i])
                else:           
                    for i in range(row_nu_1):
                        if df['Publication type'].iloc[i] in ['Journal article', 'Magazine article', 'Newspaper article']:
                            df_items = ('**'+ df['Publication type']+ '**'+ ': ' +
                                df['Title'] + ' '+ 
                                ' (by ' + '*' + df['firstName'] + '*'+ ' ' + '*' + df['lastName'] + '*' + ') ' +
                                ' (Published on: ' +df['Date published'] + ') '+
                                ' (Published in: ' + '*' + df['Journal'].iloc[i] + '*' + ') ' +
                                "[[Publication link]]" +'('+ df['Link to publication'] + ')' +'  '+
                                "[[Zotero link]]" +'('+ df['Zotero link'] + ')'
                                )                         
                            st.write('' + str(i+1) + ') ' + df_items.iloc[i])
                        else:
                            df_items = ('**'+ df['Publication type']+ '**'+ ': ' +
                                df['Title'] + ' '+ 
                                ' (by ' + '*' + df['firstName'] + '*'+ ' ' + '*' + df['lastName'] + '*' + ') ' +
                                ' (Published on: ' +df['Date published'] + ') '+
                                "[[Publication link]]" +'('+ df['Link to publication'] + ')' +'  '+
                                "[[Zotero link]]" +'('+ df['Zotero link'] + ')'
                                )   
                            st.write('' + str(i+1) + ') ' + df_items.iloc[i])
                        df_items.fillna("nan") 
                        if display2:
                            st.caption(df['Abstract'].iloc[i])

        with col2:
            with st.expander("Collections in Zotero library", expanded=False):
                @st.cache_data(ttl=300)
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
                st.caption('[Intelligence oversight and ethics](https://intelligence.streamlit.app/Intelligence_oversight_and_ethics)')
                st.caption('[Intelligence collection](https://intelligence.streamlit.app/Intelligence_collection)')
                st.caption('[Counterintelligence](https://intelligence.streamlit.app/Counterintelligence)')
                st.caption('[Covert action](https://intelligence.streamlit.app/Covert_action)')
                st.caption('[Intelligence and cybersphere](https://intelligence.streamlit.app/Intelligence_and_cybersphere)')
                st.caption('[Special collections](https://intelligence.streamlit.app/Special_collections)')

            with st.expander('Events', expanded=True):
                # Create a connection object.
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
                today = dt.date.today()
                filter = (df_gs['date']>=today)
                df_gs = df_gs.loc[filter]
                df_gs = df_gs.fillna('')
                df_gs = df_gs.head(3)
                if df_gs['event_name'].any() in ("", [], None, 0, False):
                    st.write('No upcoming event!')
                df_gs1 = ('['+ df_gs['event_name'] + ']'+ '('+ df_gs['link'] + ')'', organised by ' + '**' + df_gs['organiser'] + '**' + '. Date: ' + df_gs['date_new'] + ', Venue: ' + df_gs['venue'])
                row_nu = len(df_gs.index)
                for i in range(row_nu):
                    st.write(''+str(i+1)+') '+ df_gs1.iloc[i])
                st.write('Visit the [Events on intelligence](https://intelligence.streamlit.app/Events) page to see more!')
                
    with tab2:
        st.header('Dashboard')
        st.markdown('#### Collection theme: ' + collection_name)
        if df['Title'].any() in ("", [], None, 0, False):
            all = st.checkbox('Show all types')
            if all:
                df=df2.copy()
        types = st.multiselect('Publication type', df['Publication type'].unique(),df['Publication type'].unique(), key='original2')
        df = df[df['Publication type'].isin(types)]  #filtered_df = df[df["app"].isin(selected_options)]
        df = df.reset_index()  
        if df['Title'].any() in ("", [], None, 0, False):
            st.write('No data to visualise')
            st.stop()

        col1, col2 = st.columns(2)
        with col1:
            df_plot= df['Publication type'].value_counts()
            df_plot=df_plot.reset_index()
            df_plot=df_plot.rename(columns={'index':'Publication type','Publication type':'Count'})

            plot= df_plot
            # st.bar_chart(plot.sort_values(ascending=False), height=600, width=600, use_container_width=True)

            fig = px.pie(plot, values='Count', names='Publication type')
            fig.update_layout(title={'text':'Publications: '+collection_name, 'y':0.95, 'x':0.45, 'yanchor':'top'})
            col1.plotly_chart(fig, use_container_width = True)

        with col2:
            fig = px.bar(df_plot, x='Publication type', y='Count', color='Publication type')
            fig.update_layout(
                autosize=False,
                width=400,
                height=400,)
            fig.update_layout(title={'text':'Publications: '+collection_name, 'y':0.95, 'x':0.3, 'yanchor':'top'})
            col2.plotly_chart(fig, use_container_width = True)

        # df['Date published'] = pd.to_datetime(df['Date published'],utc=True, errors='coerce').dt.tz_convert('Europe/London')
        # df['Date published'] = df['Date published'].dt.strftime('%d-%m-%Y')
        # df['Date published'] = df['Date published'].fillna('No date')

        df['Date published'] = pd.to_datetime(df['Date published'],utc=True, errors='coerce')
        df['Date year'] = df['Date published'].dt.strftime('%Y')
        df['Date month'] = df['Date published'].dt.strftime('%Y-%m')
        df['Date year'] = df['Date year'].fillna('No date')
        df_year=df['Date year'].value_counts()
        df_year=df_year.reset_index()

        df_month = df['Date month'].value_counts()
        df_month= df_month.reset_index()
        df['Date month'] = df['Date month'].fillna('No date')
        # df['Date published'] = pd.to_datetime(df['Date published'],utc=True, errors='coerce').dt.tz_convert('Europe/London')
        # df['month'] = df['Date published'].dt.month
        # df['year'] = df['Date published'].dt.year
        # # df['year'] = df['year'].astype(int)
        # # df
        # df['Date year'] = df['Date published'].dt.strftime('%Y')
        # df['Date year'] = df['Date year'].fillna('No date')

        by_year = st.checkbox('Show by publication year')
        col1, col2 = st.columns(2)    
        with col1:
            if by_year:
                df_year=df_year.rename(columns={'index':'Publication year','Date year':'Count'})
                df_year.drop(df_year[df_year['Publication year']== 'No date'].index, inplace = True)
                df_year=df_year.sort_values(by='Publication year', ascending=True)
                fig = px.bar(df_year, x='Publication year', y='Count')
                fig.update_xaxes(tickangle=-70)
                fig.update_layout(
                    autosize=False,
                    width=400,
                    height=500,)
                fig.update_layout(title={'text':'Publications by year: '+collection_name, 'y':0.95, 'x':0.5, 'yanchor':'top'})
                col1.plotly_chart(fig, use_container_width = True)
            else:
                df_month=df_month.rename(columns={'index':'Publication month','Date month':'Count'})
                df_month.drop(df_month[df_month['Publication month']== 'No date'].index, inplace = True)
                df_month=df_month.sort_values(by='Publication month', ascending=True)
                fig = px.bar(df_month, x='Publication month', y='Count')
                fig.update_xaxes(tickangle=-70)
                fig.update_layout(
                    autosize=False,
                    width=400,
                    height=500,)
                fig.update_layout(title={'text':'Publications by month: '+collection_name, 'y':0.95, 'x':0.5, 'yanchor':'top'})
                col1.plotly_chart(fig, use_container_width = True)

        with col2:
            if by_year:
                df_year['Sum'] = df_year['Count'].cumsum()
                fig2 = px.line(df_year, x='Publication year', y='Sum')
                fig2.update_layout(title={'text':'Publications by year: '+collection_name, 'y':0.95, 'x':0.5, 'yanchor':'top'})
                fig2.update_xaxes(tickangle=-70)
                col2.plotly_chart(fig2, use_container_width = True)

            else:
                df_month['Sum'] = df_month['Count'].cumsum()
                fig2 = px.line(df_month, x='Publication month', y='Sum')
                fig2.update_layout(title={'text':'Publications by month: '+collection_name, 'y':0.95, 'x':0.5, 'yanchor':'top'})
                fig2.update_xaxes(tickangle=-70)
                col2.plotly_chart(fig2, use_container_width = True)

        col1, col2 = st.columns(2)
        with col1:
            number = st.select_slider('Select a number of publishers', options=[5,10,15,20,25,30], value=10)
            df_publisher = pd.DataFrame(df['Publisher'].value_counts())
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
            number2 = st.select_slider('Select a number of journals', options=[5,10,15,20,25,30], value=10)
            df_journal = df.loc[df['Publication type']=='Journal article']
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
                        fig.update_layout(title={'text':'Top ' + str(number2) + ' journals that publish intelligence articles)', 'y':0.95, 'x':0.4, 'yanchor':'top'})
                        col2.plotly_chart(fig, use_container_width = True)
                with st.expander('See journals'):
                    row_nu_collections = len(df_journal.index)        
                    for i in range(row_nu_collections):
                        st.caption(df_journal['Journal'].iloc[i]
                        )  

        st.write('---')
        df=df.copy()
        def clean_text (text):
            text = text.lower() # lowercasing
            text = re.sub(r'[^\w\s]', ' ', text) # this removes punctuation
            text = re.sub('[0-9_]', ' ', text) # this removes numbers
            text = re.sub('[^a-z_]', ' ', text) # removing all characters except lowercase letters
            return text
        df['clean_title'] = df['Title'].apply(clean_text)
        df['clean_abstract'] = df['Abstract'].apply(clean_text)
        df['clean_title'] = df['clean_title'].apply(lambda x: ' '.join ([w for w in x.split() if len (w)>2])) # this function removes words less than 2 words
        df['clean_abstract'] = df['clean_abstract'].apply(lambda x: ' '.join ([w for w in x.split() if len (w)>2])) # this function removes words less than 2 words

        def tokenization(text):
            text = re.split('\W+', text)
            return text
        df['token_title']=df['clean_title'].apply(tokenization)
        df['token_abstract']=df['clean_abstract'].apply(tokenization)
        stopword = nltk.corpus.stopwords.words('english')

        if collection_name=='98.1 War in Ukraine':
            SW = ['york', 'intelligence', 'security', 'pp', 'war','world', 'article', 'twitter', 'invasion',
                'ukraine', 'russian', 'ukrainian', 'russia', 'could', 'vladimir',
                'new', 'isbn', 'book', 'also', 'yet', 'matter', 'erratum', 'commentary', 'studies',
                'volume', 'paper', 'study', 'question', 'editorial', 'welcome', 'introduction', 'editorial', 'reader',
                'university', 'followed', 'particular', 'based', 'press', 'examine', 'show', 'may', 'result', 'explore',
                'examines', 'become', 'used', 'journal', 'london', 'review']
        else:
            SW = ['york', 'intelligence', 'security', 'pp', 'war','world', 'article', 'twitter',
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
            plt.title('Top words in title (collection: ' +collection_name+')')
            plt.imshow(wordcloud)
            plt.axis("off")
            plt.show()
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.pyplot() 
        else:
            st.warning('Please bear in mind that not all items listed in this bibliography have an abstract. Therefore, this wordcloud should not be considered as authoritative.')
            df_list_abstract = [item for sublist in listdf_abstract for item in sublist]
            string = pd.Series(df_list_abstract).str.cat(sep=' ')
            wordcloud_texts = string
            wordcloud_texts_str = str(wordcloud_texts)
            wordcloud = WordCloud(stopwords=stopword, width=1500, height=750, background_color='white', collocations=False, colormap='magma').generate(wordcloud_texts_str)
            plt.figure(figsize=(20,8))
            plt.axis('off')
            plt.title('Top words in abstract (collection: ' +collection_name+')')
            plt.imshow(wordcloud)
            plt.axis("off")
            plt.show()
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.pyplot() 

    with tab3:
        df=df2.copy()
        row_nu_1 = len(df.index)
        df = df.reset_index()
        df = df.drop(['index'], axis=1)
        if row_nu_1 >5:
            df=df.sample(n=5)
            row_nu_1= len(df.index)
        df = df.reset_index()
        st.info('It may take some time for the app to bring new results as it searches the entire database.')
        if st.button('🔀 Suggest new sources'):
            df=df2.copy()
            row_nu_1 = len(df.index)
            df = df.reset_index()
            df = df.drop(['index'], axis=1)
            if row_nu_1 >5:
                df=df.sample(n=5)
                row_nu_1= len(df.index)
            df = df.reset_index()
        if df['FirstName2'].any() in ("", [], None, 0, False):
            # st.write('no author')
            df['firstName'] = 'null'
            df['lastName'] = 'null'

            df_items = ('**'+ df['Publication type']+ '**'+ ': ' +
                df['Title'] + ' '+ 
                ' (by ' + '*' + df['firstName'] + '*'+ ' ' + '*' + df['lastName'] + '*' + ') ' + 
                "[[Publication link]]" +'('+ df['Link to publication'] + ')' +'  '+
                "[[Zotero link]]" +'('+ df['Zotero link'] + ')' +
                ' (Published on: ' +df['Date published'] + ')'
                )
        else:
                # st.write('author entered')
                ## This section is for displaying the first author details but it doesn't work for now because of json normalization error.
                df_fa = df['FirstName2']
                df_fa = pd.DataFrame(df_fa.tolist())
                df_fa = df_fa[0]
                df_fa = df_fa.apply(lambda x: {} if pd.isna(x) else x) # https://stackoverflow.com/questions/44050853/pandas-json-normalize-and-null-values-in-json
                df_new = pd.json_normalize(df_fa, errors='ignore') 
                df = pd.concat([df, df_new], axis=1)
                df['firstName'] = df['firstName'].fillna('null')
                df['lastName'] = df['lastName'].fillna('null')    
                df_items = ('**'+ df['Publication type']+ '**'+ ': ' +
                            df['Title'] + ' '+ 
                            ' (by ' + '*' + df['firstName'] + '*'+ ' ' + '*' + df['lastName'] + '*' + ') ' + # IT CANNOT READ THE NAN VALUES
                            "[[Publication link]]" +'('+ df['Link to publication'] + ')' +'  '+
                            "[[Zotero link]]" +'('+ df['Zotero link'] + ')' +
                            ' (Published on: ' +df['Date published'] + ')'
                            )
        for i in range(row_nu_1):
            st.write(''+str(i+1)+') ' +df_items.iloc[i])
            df_items.fillna("nan") 
            if display2:
                st.caption(df['Abstract'].iloc[i])
                    
    components.html(
    """
    <a rel="license" href="http://creativecommons.org/licenses/by/4.0/"><img alt="Creative Commons Licence" style="border-width:0" 
    src="https://i.creativecommons.org/l/by/4.0/80x15.png" /></a><br />
    © 2022 All rights reserved. This website is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">Creative Commons Attribution 4.0 International License</a>.
    """
    )
