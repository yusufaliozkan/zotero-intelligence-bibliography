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
import random
from authors_dict import df_authors, name_replacements

st.set_page_config(layout = "wide", 
                    page_title='Intelligence studies network',
                    page_icon="https://images.pexels.com/photos/315918/pexels-photo-315918.png",
                    initial_sidebar_state="auto") 

st.title("Intelligence history")

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

    # zot = zotero.Zotero(library_id, library_type)

    # @st.cache_data(ttl=300)
    # def zotero_collections(library_id, library_type):
    #     collections = zot.collections()
    #     data2=[]
    #     columns2 = ['Key','Name', 'Link']
    #     for item in collections:
    #         data2.append((item['data']['key'], item['data']['name'], item['links']['alternate']['href']))
    #     pd.set_option('display.max_colwidth', None)
    #     df_collections = pd.DataFrame(data2, columns=columns2)
    #     return df_collections
    # df_collections = zotero_collections(library_id, library_type)

    df_collections = pd.read_csv('all_items_duplicated.csv')
    df_collections = df_collections[~df_collections['Collection_Name'].str.contains('01.98')]
    # df_collections = df_collections[df_collections['Collection_Name'] != '01 Intelligence history']


    df_collections = df_collections.sort_values(by='Collection_Name')
    df_collections=df_collections[df_collections['Collection_Name'].str.contains("98.")]
    df_collections = df_collections.iloc[1: , :]

    st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)

    container = st.container()

    tab1, tab2 = st.tabs(['📑 Publications', '📊 Dashboard'])
    with tab1:
        col1, col2 = st.columns([5,1.6])
        with col1:
            unique_collections = list(df_collections['Collection_Name'].unique()) 
            radio = container.radio('Select a collection', unique_collections)
            # collection_name = st.selectbox('Select a collection:', clist)
            collection_name = radio
            df_collections = df_collections.loc[df_collections['Collection_Name']==collection_name]
            pd.set_option('display.max_colwidth', None)

            df_collections['Date published'] = pd.to_datetime(df_collections['Date published'],utc=True, errors='coerce').dt.tz_convert('Europe/London')
            df_collections['Date published'] = df_collections['Date published'].dt.strftime('%Y-%m-%d')
            df_collections['Date published'] = df_collections['Date published'].fillna('')
            df_collections['No date flag'] = df_collections['Date published'].isnull().astype(np.uint8)
            df_collections = df_collections.sort_values(by=['No date flag', 'Date published'], ascending=[True, True])
            df_collections = df_collections.sort_values(by=['Date published'], ascending=False)
            df_collections = df_collections.reset_index(drop=True)

            publications_by_type = df_collections['Publication type'].value_counts()
            collection_link = df_collections[df_collections['Collection_Name'] == collection_name]['Collection_Link'].iloc[0]

            st.markdown('#### Collection theme: ' + collection_name)
            st.write(f"See the collection in [Zotero]({collection_link})")
            types = st.multiselect('Publication type', df_collections['Publication type'].unique(),df_collections['Publication type'].unique(), key='original')
            df_collections = df_collections[df_collections['Publication type'].isin(types)]
            df_collections = df_collections.reset_index(drop=True)
            df_download = df_collections[['Publication type','Title','FirstName2','Abstract','Date published','Publisher','Journal','Link to publication','Zotero link']]
            df_download = df_download.reset_index(drop=True)
            def convert_df(df_download):
                return df_download.to_csv(index=False).encode('utf-8-sig')
            csv = convert_df(df_download)
            today = datetime.date.today().isoformat()
            num_items_collections = len(df_collections)
            breakdown_string = ', '.join([f"{key}: {value}" for key, value in publications_by_type.items()])
            st.write(f"**{num_items_collections}** sources found ({breakdown_string})")
            a = f'{collection_name}_{today}'
            st.download_button('💾 Download the collection', csv, (a+'.csv'), mime="text/csv", key='download-csv-4')

            with st.expander('Click to expand', expanded=True):
                def format_entry(row):
                    publication_type = str(row['Publication type']) if pd.notnull(row['Publication type']) else ''
                    title = str(row['Title']) if pd.notnull(row['Title']) else ''
                    authors = str(row['FirstName2'])
                    date_published = str(row['Date published']) if pd.notnull(row['Date published']) else ''
                    link_to_publication = str(row['Link to publication']) if pd.notnull(row['Link to publication']) else ''
                    zotero_link = str(row['Zotero link']) if pd.notnull(row['Zotero link']) else ''
                    published_by_or_in = ''
                    published_source = ''

                    if publication_type == 'Journal article':
                        published_by_or_in = 'Published in'
                        published_source = str(row['Journal']) if pd.notnull(row['Journal']) else ''
                    elif publication_type == 'Book':
                        published_by_or_in = 'Published by'
                        published_source = str(row['Publisher']) if pd.notnull(row['Publisher']) else ''
                    else:
                        # For other types, leave the fields empty
                        published_by_or_in = ''
                        published_source = ''

                    return (
                        '**' + publication_type + '**' + ': ' +
                        title + ' ' +
                        '(by ' + '*' + authors + '*' + ') ' +
                        '(Publication date: ' + str(date_published) + ') ' +
                        ('(' + published_by_or_in + ': ' + '*' + published_source + '*' + ') ' if published_by_or_in else '') +
                        '[[Publication link]](' + link_to_publication + ') ' +
                        '[[Zotero link]](' + zotero_link + ')'
                    )

                articles_list = []  # Store articles in a list
                for index, row in df_collections.iterrows():
                    formatted_entry = format_entry(row)  # Assuming format_entry() is a function formatting each row
                    articles_list.append(formatted_entry)                     
                
                for index, row in df_collections.iterrows():
                    publication_type = row['Publication type']
                    title = row['Title']
                    authors = row['FirstName2']
                    date_published = row['Date published']
                    link_to_publication = row['Link to publication']
                    zotero_link = row['Zotero link']

                    if publication_type == 'Journal article':
                        published_by_or_in = 'Published in'
                        published_source = str(row['Journal']) if pd.notnull(row['Journal']) else ''
                    elif publication_type == 'Book':
                        published_by_or_in = 'Published by'
                        published_source = str(row['Publisher']) if pd.notnull(row['Publisher']) else ''
                    else:
                        published_by_or_in = ''
                        published_source = ''

                    formatted_entry = (
                        '**' + str(publication_type) + '**' + ': ' +
                        str(title) + ' ' +
                        '(by ' + '*' + str(authors) + '*' + ') ' +
                        '(Publication date: ' + str(date_published) + ') ' +
                        ('(' + published_by_or_in + ': ' + '*' + str(published_source) + '*' + ') ' if published_by_or_in else '') +
                        '[[Publication link]](' + str(link_to_publication) + ') ' +
                        '[[Zotero link]](' + str(zotero_link) + ')'
                    )
                sort_by_type = st.checkbox('Sort by publication type', key='type')
                display2 = st.checkbox('Display abstracts')

                if sort_by_type:
                    df_collections = df_collections.sort_values(by=['Publication type'], ascending=True)
                    current_type = None
                    count_by_type = {}
                    for index, row in df_collections.iterrows():
                        if row['Publication type'] != current_type:
                            current_type = row['Publication type']
                            st.subheader(current_type)
                            count_by_type[current_type] = 1
                        formatted_entry = format_entry(row)
                        st.write(f"{count_by_type[current_type]}) {formatted_entry}")
                        count_by_type[current_type] += 1
                        if display2:
                            st.caption(row['Abstract'])
                else:
                    count = 1
                    for index, row in df_collections.iterrows():
                        formatted_entry = format_entry(row)
                        st.write(f"{count}) {formatted_entry}")
                        count += 1
                        if display2:
                            st.caption(row['Abstract'])

#UNTIL HERE
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

        if df_collections['Title'].any() in ("", [], None, 0, False):
            all = st.checkbox('Show all types')
            if all:
                df=df_collections.copy()
        types = st.multiselect('Publication type', df_collections['Publication type'].unique(),df_collections['Publication type'].unique(), key='original2')
        df_collections = df_collections[df_collections['Publication type'].isin(types)]  #filtered_df = df[df["app"].isin(selected_options)]
        df_collections = df_collections.reset_index()
        
        if df_collections['Title'].any() in ("", [], None, 0, False):
            st.write('No data to visualise')
            st.stop()

        col1, col2 = st.columns(2)
        with col1:
            df_plot= df_collections['Publication type'].value_counts()
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

        df_collections['Date published'] = pd.to_datetime(df_collections['Date published'],utc=True, errors='coerce').dt.tz_convert('Europe/London')
        df_collections['Date year'] = df_collections['Date published'].dt.strftime('%Y')
        df_collections['Date year'] = df_collections['Date year'].fillna('No date')
        df_year=df_collections['Date year'].value_counts()
        df_year=df_year.reset_index()

        col1, col2 = st.columns(2)
        with col1:
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

        with col2:
            df_collections['Author_name'] = df_collections['FirstName2'].apply(lambda x: x.split(', ') if isinstance(x, str) and x else x)
            df_collections = df_collections.explode('Author_name')
            df_collections.reset_index(drop=True, inplace=True)
            df_collections = df_collections.loc[df_collections['Collection_Name']==collection_name]
            df_collections
            max_authors = len(df_collections['Author_name'].unique())
            num_authors = st.slider('Select number of authors to display:', 1, min(50, max_authors), 20)
            
            # Adding a multiselect widget for publication types
            selected_types = st.multiselect('Select publication types:', df_collections['Publication type'].unique(), default=df_collections['Publication type'].unique())
            
            # Filtering data based on selected publication types
            filtered_authors = df_collections[df_collections['Publication type'].isin(selected_types)]
            
            if len(selected_types) == 0:
                st.write('No results to display')
            else:
                publications_by_author = filtered_authors['Author_name'].value_counts().head(num_authors)
                fig = px.bar(publications_by_author, x=publications_by_author.index, y=publications_by_author.values)
                fig.update_layout(
                    title=f'Top {num_authors} Authors by Publication Count',
                    xaxis_title='Author',
                    yaxis_title='Number of Publications',
                    xaxis_tickangle=-45,
                )
                col2.plotly_chart(fig)
            df_collections = df_collections.drop_duplicates(subset='Zotero link')
            df_collections = df_collections.reset_index(drop=True)

        col1, col2 = st.columns(2)
        with col1:
            number = st.select_slider('Select a number of publishers', options=[5,10,15,20,25,30], value=10)
            df_publisher = pd.DataFrame(df_collections['Publisher'].value_counts())
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
            df_journal = df_collections.loc[df_collections['Publication type']=='Journal article']
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
        df=df_collections.copy()

        def clean_text(text):
            if pd.isnull(text):  # Check if the value is NaN
                return ''  # Return an empty string or handle it based on your requirement
            text = str(text)  # Convert to string to ensure string methods can be applied
            text = text.lower()  # Lowercasing
            text = re.sub(r'[^\w\s]', ' ', text)  # Removes punctuation
            text = re.sub('[0-9_]', ' ', text)  # Removes numbers
            text = re.sub('[^a-z_]', ' ', text)  # Removes all characters except lowercase letters
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

        if collection_name=='01.6 WW2 (Second World War)':
            SW = ['york', 'intelligence', 'security', 'pp', 'war','world', 'article', 'twitter', 'thesis', 'chapter', 'second',
                'new', 'isbn', 'book', 'also', 'yet', 'matter', 'erratum', 'commentary', 'studies', 'effective', 'important', 'good', 'put',
                'argued', 'mean', 'one', 'allow', 'contrary', 'investigates', 'could', 'history',
                'volume', 'paper', 'study', 'question', 'editorial', 'welcome', 'introduction', 'editorial', 'reader',
                'university', 'followed', 'particular', 'based', 'press', 'examine', 'show', 'may', 'result', 'explore',
                'examines', 'become', 'used', 'journal', 'london', 'review']
        if collection_name=='01.4 WW1 (First Wold War)':
            SW = ['york', 'intelligence', 'security', 'pp', 'war','world', 'article', 'twitter', 'thesis', 'chapter',
                'new', 'first', 'isbn', 'book', 'also', 'yet', 'matter', 'erratum', 'commentary', 'studies', 'effective', 'important', 'good', 'put',
                'argued', 'mean', 'one', 'allow', 'contrary', 'investigates', 'could', 'history',
                'volume', 'paper', 'study', 'question', 'editorial', 'welcome', 'introduction', 'editorial', 'reader',
                'university', 'followed', 'particular', 'based', 'press', 'examine', 'show', 'may', 'result', 'explore',
                'examines', 'become', 'used', 'journal', 'london', 'review']        
        else:
            SW = ['york', 'intelligence', 'security', 'pp', 'war','world', 'article', 'twitter', 'thesis', 'chapter',
                'new', 'isbn', 'book', 'also', 'yet', 'matter', 'erratum', 'commentary', 'studies', 'effective', 'important', 'good', 'put',
                'argued', 'mean', 'one', 'allow', 'contrary', 'investigates', 'could', 'history',
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

    components.html(
    """
    <a rel="license" href="http://creativecommons.org/licenses/by/4.0/"><img alt="Creative Commons Licence" style="border-width:0" 
    src="https://i.creativecommons.org/l/by/4.0/80x15.png" /></a><br />
    © 2022 All rights reserved. This website is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">Creative Commons Attribution 4.0 International License</a>.
    """
    )