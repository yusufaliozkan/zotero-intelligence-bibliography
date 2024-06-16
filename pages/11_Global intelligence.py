from pyzotero import zotero
import pandas as pd
import streamlit as st
# from IPython.display import HTML
import streamlit.components.v1 as components
import numpy as np
# import altair as alt
# from pandas.io.json import json_normalize
import datetime
import plotly.express as px
import re
import matplotlib.pyplot as plt
import nltk
nltk.download('all')
from nltk.corpus import stopwords
nltk.download('stopwords')
from wordcloud import WordCloud
#import datetime as dt     
# import random
from authors_dict import df_authors, name_replacements
from countries_dict import country_names, replacements, df_countries, df_continent
from sidebar_content import sidebar_content
import time
from format_entry import format_entry
from events import evens_conferences

st.set_page_config(layout = "wide", 
                    page_title='Intelligence studies network',
                    page_icon="https://images.pexels.com/photos/315918/pexels-photo-315918.png",
                    initial_sidebar_state="auto") 
st.title("Global intelligence") 

with st.spinner('Retrieving data & updating dashboard...'):
    sidebar_content()

    col1, col2 = st.columns([1,3])
    with col1:
        container_metric = st.container()
    with col2: 
        
        @st.cache_data(ttl=100)
        def load_data():
            df_collections = pd.read_csv('all_items_duplicated.csv')
            df_collections = df_collections.sort_values(by='Collection_Name')
            return df_collections

        df_collections = load_data()
        df_collections = df_collections[df_collections['Collection_Name'].str.contains("14.")]

        st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
        
        # container = st.container()

        collection_name = df_collections['Collection_Name'].iloc[0]
        pd.set_option('display.max_colwidth', None)

        # df_collections['Date published'] = pd.to_datetime(df_collections['Date published'],utc=True, errors='coerce').dt.tz_convert('Europe/London')
        df_collections['Date published'] = (
            df_collections['Date published']
            .str.strip()
            .apply(lambda x: pd.to_datetime(x, utc=True, errors='coerce').tz_convert('Europe/London'))
        )
        df_collections['Date published'] = df_collections['Date published'].dt.strftime('%Y-%m-%d')
        df_collections['Date published'] = df_collections['Date published'].fillna('')
        df_collections['No date flag'] = df_collections['Date published'].isnull().astype(np.uint8)
        df_collections = df_collections.sort_values(by=['No date flag', 'Date published'], ascending=[True, True])
        df_collections = df_collections.sort_values(by=['Date published'], ascending=False)
        df_collections = df_collections.reset_index(drop=True)

        publications_by_type = df_collections['Publication type'].value_counts()
        collection_link = df_collections[df_collections['Collection_Name'] == collection_name]['Collection_Link'].iloc[0] 

        # st.markdown('#### Collection theme: ' + collection_name)
        with st.popover('More details'):
            st.info(f'''
            This collection lists academic sources that are **non-UK/US** on intelligence. 
            
            Pick up a country from the drop down menu too publications.

            See the collection in [Zotero]({collection_link}) from which you can easily generate citations.
            ''')
           
            df_countries_chart = df_countries.copy()
            df_continent = df_continent.copy()
            df_continent_chart = df_continent.copy() 

            unique_items_count = df_countries_chart['Country'].nunique()
            num_items_collections = len(df_collections)
            true_count = df_collections[df_collections['Publication type']=='Journal article']['OA status'].sum()
            total_count = len(df_collections[df_collections['Publication type']=='Journal article'])
            if total_count == 0:
                oa_ratio = 0.0
            else:
                oa_ratio = true_count / total_count * 100
        
            def split_and_expand(authors):
                # Ensure the input is a string
                if isinstance(authors, str):
                    # Split by comma and strip whitespace
                    split_authors = [author.strip() for author in authors.split(',')]
                    return pd.Series(split_authors)
                else:
                    # Return the original author if it's not a string
                    return pd.Series([authors])
            expanded_authors = df_collections['FirstName2'].apply(split_and_expand).stack().reset_index(level=1, drop=True)
            expanded_authors = expanded_authors.reset_index(name='Author')
            author_no = len(expanded_authors)
            if author_no == 0:
                author_pub_ratio=0.0
            else:
                author_pub_ratio = round(author_no/num_items_collections, 2)

            citation_count = df_collections['Citation'].sum()
            st.metric(label='Number of country', value=unique_items_count-1)
            st.metric(label="Number of citations", value=int(citation_count), help='Journal articles only')
            st.metric(label="Open access coverage", value=f'{int(oa_ratio)}%', help='Journal articles only')
            st.metric(label='Number of authors', value=int(author_no))
            st.metric(label='Author/publication ratio', value=author_pub_ratio, help='The average author number per publication')

            container_metric = container_metric.metric(label='Number of items', value=num_items_collections, help=f'sources found for **{unique_items_count-1}** countries.')

    df_countries['Date published'] = ( 
        df_countries['Date published']
        .str.strip()
        .apply(lambda x: pd.to_datetime(x, utc=True, errors='coerce').tz_convert('Europe/London'))
    )
    # df_countries['Date published'] = pd.to_datetime(df_countries['Date published'],utc=True, errors='coerce').dt.tz_convert('Europe/London')
    df_countries['Date published'] = df_countries['Date published'].dt.strftime('%Y-%m-%d')
    df_countries['Date published'] = df_countries['Date published'].fillna('')
    df_countries['No date flag'] = df_countries['Date published'].isnull().astype(np.uint8)
    df_countries = df_countries.sort_values(by=['No date flag', 'Date published'], ascending=[True, True])
    df_countries = df_countries.sort_values(by=['Date published'], ascending=False)
    df_countries = df_countries.drop_duplicates(subset=['Country', 'Zotero link'])
    df_countries = df_countries.reset_index(drop=True)
    unique_countries = [''] + ['All Countries'] + sorted(df_countries['Country'].unique())

    # Function to update query_params based on selected country
    def update_params():
        if st.session_state.qp:
            st.query_params.from_dict({'country': st.session_state.qp})

    # Retrieve query_params and initialize selected_country
    query_params = st.query_params
    selected_country = query_params.get('country', '')  # Use .get() to handle None gracefully

    # Calculate selected_country_index to set the initial index of the selectbox
    selected_country_index = unique_countries.index(selected_country) if selected_country in unique_countries else 0

    # Create selectbox to choose a country
    selected_country = st.selectbox('**Select a Country**', unique_countries, index=selected_country_index, on_change=update_params, key='qp')

    # Query_params handling based on selected country
    ix = 0
    if selected_country:
        try:
            # Get the index of selected_country in unique_countries
            ix = unique_countries.index(selected_country)
        except ValueError:
            pass

    if selected_country=='':
        query_params.clear()

    number_of_pub = df_countries[df_countries['Country'] == selected_country]
    publications_count = len(number_of_pub)

    # Filter the DataFrame based on the selected country
    df_countries = df_countries[df_countries['Country'] == selected_country]

    st.divider()

    st.subheader(f"{selected_country}")
    if selected_country!='':
        col1, col2, col3 = st.columns([2,2,2])
        with col1:
            container_metric_2 = st.container()
        with col2:
            with st.popover('More metrics'):
                container_citation_2 = st.container()
                container_oa = st.container()
                container_type = st.container()
                container_author_no = st.container()
                container_country = st.container()
                container_author_pub_ratio = st.container()
        with col3:
            with st.popover('Filters and more'):
                col31, col32, col33 = st.columns(3)
                with col31:
                    container_abstract = st.container()
                with col32:
                    container_cited_items = st.container()
                with col33:
                    table_view = st.checkbox('See results in table')
                container_pub_types = st.container()
                container_download = st.container()

    tab1, tab2 = st.tabs(['📑 Publications', '📊 Dashboard'])
    with tab1:
        col1, col2 = st.columns([5,1.6])
        with col1:
            # Display the filtered DataFrame
            # THIS WAS THE PLACE WHERE FORMAT_ENTRY WAS LOCATED
            if not selected_country or selected_country=="":
                st.warning('Please select a country from the dropdown menu above to see publications.')
                st.divider()
            
            elif selected_country == 'All Countries':
                with st.expander('Click to expand', expanded=True):
                    only_citation = container_cited_items.checkbox('Show cited items only')
                    if only_citation:
                        df_collections = df_collections[(df_collections['Citation'].notna()) & (df_collections['Citation'] != 0)]
                        df_countries = df_countries[(df_countries['Citation'].notna()) & (df_countries['Citation'] != 0)]

                    types = container_pub_types.multiselect('Publication type', df_collections['Publication type'].unique(),df_collections['Publication type'].unique(), key='original')
                    df_collections = df_collections[df_collections['Publication type'].isin(types)]
                    df_collections = df_collections.reset_index(drop=True)
                    df_collections['FirstName2'] = df_collections['FirstName2'].map(name_replacements).fillna(df_collections['FirstName2'])
                    df_download = df_collections[['Publication type','Title','FirstName2','Abstract','Date published','Publisher','Journal','Link to publication','Zotero link']]
                    df_download = df_download.reset_index(drop=True)
                    def convert_df(df_download):
                        return df_download.to_csv(index=False).encode('utf-8-sig')
                    csv = convert_df(df_download)
                    today = datetime.date.today().isoformat()
                    num_items_collections = len(df_collections)
                    breakdown_string = ', '.join([f"{key}: {value}" for key, value in publications_by_type.items()])
                    a = f'{collection_name}_{today}'
                    container_download.download_button('💾 Download the collection', csv, (a+'.csv'), mime="text/csv", key='download-csv-4')

                    # st.write(f"**{num_items_collections}** sources found ({breakdown_string})")
                    true_count = df_collections[df_collections['Publication type']=='Journal article']['OA status'].sum()
                    total_count = len(df_collections[df_collections['Publication type']=='Journal article'])
                    if total_count == 0:
                        oa_ratio = 0.0
                    else:
                        oa_ratio = true_count / total_count * 100

                    citation_count = df_collections['Citation'].sum()
                    def split_and_expand(authors):
                        # Ensure the input is a string
                        if isinstance(authors, str):
                            # Split by comma and strip whitespace
                            split_authors = [author.strip() for author in authors.split(',')]
                            return pd.Series(split_authors)
                        else:
                            # Return the original author if it's not a string
                            return pd.Series([authors])
                    if len(df_collections)==0:
                        author_no=0
                    else:       
                        expanded_authors = df_collections['FirstName2'].apply(split_and_expand).stack().reset_index(level=1, drop=True)
                        expanded_authors = expanded_authors.reset_index(name='Author')
                        author_no = len(expanded_authors)
                    if author_no == 0:
                        author_pub_ratio=0.0
                    else:
                        author_pub_ratio = round(author_no/num_items_collections, 2)
                    # st.write(f'Number of citations: **{int(citation_count)}**, Open access coverage (journal articles only): **{int(oa_ratio)}%**')
                    item_type_no = df_collections['Publication type'].nunique()

                    container_metric_2.metric('Number of publications', value=num_items_collections, help=breakdown_string)
                    container_citation_2.metric(label="Number of citations", value=int(citation_count))
                    container_oa.metric(label="Open access coverage", value=f'{int(oa_ratio)}%', help='Journal articles only')
                    container_type.metric(label='Number of publication types', value=int(item_type_no))
                    container_author_no.metric(label='Number of authors', value=int(author_no))
                    container_country.metric(label='Number of country', value=unique_items_count-1)
                    container_author_pub_ratio.metric(label='Author/publication ratio', value=author_pub_ratio, help='The average author number per publication')

                    # THIS WAS THE PLACE WHERE FORMAT_ENTRY WAS LOCATED
                    sort_by = st.radio('Sort by:', ('Publication date :arrow_down:', 'Publication type',  'Citation'))
                    display2 = container_abstract.checkbox('Display abstracts', key='type_count2')

                    if not table_view:
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

                        if sort_by == 'Publication date :arrow_down:' or df_collections['Citation'].sum() == 0:
                            count = 1
                            for index, row in df_collections.iterrows():
                                formatted_entry = format_entry(row)
                                st.write(f"{count}) {formatted_entry}")
                                count += 1
                                if display2:
                                    st.caption(row['Abstract']) 
                        elif sort_by == 'Publication type' or df_collections['Citation'].sum() == 0:
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
                            df_collections = df_collections.sort_values(by=['Citation'], ascending=False)
                            count = 1
                            for index, row in df_collections.iterrows():
                                formatted_entry = format_entry(row)
                                st.write(f"{count}) {formatted_entry}")
                                count += 1
                                if display2:
                                    st.caption(row['Abstract']) 
                    else:
                        df_table_view = df_collections[['Publication type','Title','Date published','FirstName2', 'Abstract','Publisher','Journal', 'Citation', 'Collection_Name','Link to publication','Zotero link']]
                        df_table_view = df_table_view.rename(columns={'FirstName2':'Author(s)','Collection_Name':'Collection','Link to publication':'Publication link'})
                        if sort_by == 'Publication type':
                            df_table_view = df_table_view.sort_values(by=['Publication type'], ascending=True)
                            df_table_view = df_table_view.reset_index(drop=True)
                            df_table_view
                        elif sort_by == 'Citation':
                            df_table_view = df_table_view.sort_values(by=['Citation'], ascending=False)
                            df_table_view = df_table_view.reset_index(drop=True)
                            df_table_view
                        else:
                            df_table_view

            else:
                with st.expander('Click to expand', expanded=True):
                    only_citation = container_cited_items.checkbox('Show cited items only')
                    if only_citation:
                        df_countries = df_countries[(df_countries['Citation'].notna()) & (df_countries['Citation'] != 0)]
                        df_countries = df_countries[(df_countries['Citation'].notna()) & (df_countries['Citation'] != 0)]
                    
                    types = container_pub_types.multiselect('Publication type', df_countries['Publication type'].unique(),df_countries['Publication type'].unique(), key='original_4')
                    df_countries = df_countries[df_countries['Publication type'].isin(types)]
                    df_countries = df_countries.reset_index(drop=True)
                    df_download = df_countries[['Publication type','Title','FirstName2','Abstract','Date published','Publisher','Journal','Link to publication','Zotero link']]
                    df_download = df_download.reset_index(drop=True)
                    def convert_df(df_download):
                        return df_download.to_csv(index=False).encode('utf-8-sig')
                    csv = convert_df(df_download)
                    today = datetime.date.today().isoformat()
                    a = f'{selected_country}_{today}'
                    container_download.download_button('💾 Download items', csv, (a+'.csv'), mime="text/csv", key='download-csv-5')

                    publications_by_type_country = df_countries['Publication type'].value_counts()
                    num_items_collections = len(df_countries)
                    breakdown_string = ', '.join([f"{key}: {value}" for key, value in publications_by_type_country.items()])                    

                    sort_by = st.radio('Sort by:', ('Publication date :arrow_down:', 'Publication type',  'Citation'))
                    display2 = container_abstract.checkbox('Display abstracts', key='type_country_2')

                    articles_list = []  # Store articles in a list
                    st.write(f"**{num_items_collections}** sources found ({breakdown_string})")

                    true_count = df_countries[df_countries['Publication type']=='Journal article']['OA status'].sum()
                    total_count = len(df_countries[df_countries['Publication type']=='Journal article'])
                    if total_count == 0:
                        oa_ratio = 0.0
                    else:
                        oa_ratio = true_count / total_count * 100

                    def split_and_expand(authors):
                        # Ensure the input is a string
                        if isinstance(authors, str):
                            # Split by comma and strip whitespace
                            split_authors = [author.strip() for author in authors.split(',')]
                            return pd.Series(split_authors)
                        else:
                            # Return the original author if it's not a string
                            return pd.Series([authors])
                    if len(df_countries)==0:
                        author_no=0
                    else:
                        expanded_authors = df_countries['FirstName2'].apply(split_and_expand).stack().reset_index(level=1, drop=True)
                        expanded_authors = expanded_authors.reset_index(name='Author')
                        author_no = len(expanded_authors)
                    if author_no == 0:
                        author_pub_ratio=0.0
                    else:
                        author_pub_ratio = round(author_no/num_items_collections, 2)

                    citation_count = df_countries['Citation'].sum()
                    item_type_no = df_countries['Publication type'].nunique()
                    st.write(f'Number of citations: **{int(citation_count)}**, Open access coverage (journal articles only): **{int(oa_ratio)}%**')

                    container_metric_2.metric('Number of publications', value=num_items_collections)
                    container_citation_2.metric(label="Number of citations", value=int(citation_count))
                    container_oa.metric(label="Open access coverage", value=f'{int(oa_ratio)}%', help='Journal articles only')
                    container_type.metric(label='Number of publication types', value=int(item_type_no))
                    container_author_no.metric(label='Number of authors', value=int(author_no))
                    container_country.metric(label='Number of country', value=unique_items_count-1)
                    container_author_pub_ratio.metric(label='Author/publication ratio', value=author_pub_ratio, help='The average author number per publication')

                        
                    if not table_view:      
                        for index, row in df_countries.iterrows():
                            formatted_entry = format_entry(row)  # Assuming format_entry() is a function formatting each row
                            articles_list.append(formatted_entry)        
                        
                        for index, row in df_countries.iterrows():
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

                        if sort_by == 'Publication date :arrow_down:' or df_countries['Citation'].sum() == 0:
                            count = 1
                            for index, row in df_countries.iterrows():
                                formatted_entry = format_entry(row)
                                st.write(f"{count}) {formatted_entry}")
                                count += 1
                                if display2:
                                    st.caption(row['Abstract']) 
                        elif sort_by == 'Publication type' or df_countries['Citation'].sum() == 0:
                            df_countries = df_countries.sort_values(by=['Publication type'], ascending=True)
                            current_type = None
                            count_by_type = {}
                            for index, row in df_countries.iterrows():
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
                            df_countries = df_countries.sort_values(by=['Citation'], ascending=False)
                            count = 1
                            for index, row in df_countries.iterrows():
                                formatted_entry = format_entry(row)
                                st.write(f"{count}) {formatted_entry}")
                                count += 1
                                if display2:
                                    st.caption(row['Abstract']) 
                    else:
                        df_table_view = df_countries[['Publication type','Title','Date published','FirstName2', 'Abstract','Publisher','Journal', 'Citation', 'Collection_Name','Link to publication','Zotero link']]
                        df_table_view = df_table_view.rename(columns={'FirstName2':'Author(s)','Collection_Name':'Collection','Link to publication':'Publication link'})
                        if sort_by == 'Publication type':
                            df_table_view = df_table_view.sort_values(by=['Publication type'], ascending=True)
                            df_table_view = df_table_view.reset_index(drop=True)
                            df_table_view
                        elif sort_by == 'Citation':
                            df_table_view = df_table_view.sort_values(by=['Citation'], ascending=False)
                            df_table_view = df_table_view.reset_index(drop=True)
                            df_table_view
                        else:
                            df_table_view                                 

            df_continent['Date published'] = pd.to_datetime(df_continent['Date published'],utc=True, errors='coerce').dt.tz_convert('Europe/London')
            df_continent['Date published'] = df_continent['Date published'].dt.strftime('%Y-%m-%d')
            df_continent['Date published'] = df_continent['Date published'].fillna('')
            df_continent['No date flag'] = df_continent['Date published'].isnull().astype(np.uint8)
            df_continent = df_continent.sort_values(by=['No date flag', 'Date published'], ascending=[True, True])
            df_continent = df_continent.sort_values(by=['Date published'], ascending=False)
            df_continent = df_continent.drop_duplicates(subset=['Continent', 'Zotero link'])
            df_continent = df_continent.reset_index(drop=True)
            unique_continents = sorted(df_continent['Continent'].unique())
            unique_continents =  [''] + list(unique_continents)  # Added 'All Countries' option


            # # Function to update query_params based on selected country
            # def update_params2():
            #     if st.session_state.qt:
            #         st.query_params.from_dict({'country': st.session_state.qt})

            # # Retrieve query_params and initialize selected_country
            # query_params2 = st.query_params
            # selected_continent = query_params2.get('continent', '')  # Use .get() to handle None gracefully

            # # Calculate selected_country_index to set the initial index of the selectbox
            # selected_continent_index = unique_continents.index(selected_continent) if selected_continent in unique_continents else 0

            # # Create selectbox to choose a country
            # selected_continent = st.selectbox('Select a Continent', unique_continents, index=selected_continent_index, on_change=update_params2, key='qt')

            # # Query_params handling based on selected country
            # ix = 0
            # if selected_continent:
            #     try:
            #         # Get the index of selected_country in unique_countries
            #         ix = unique_continents.index(selected_continent)
            #     except ValueError:
            #         pass

            # number_of_pub = df_continent[df_continent['Continent'] == selected_continent]

            # publications_count = len(number_of_pub)

            # number_of_pub_con = df_continent[df_continent['Continent'] == selected_continent]
            # publications_count_con = len(number_of_pub_con)

            # df_continent = df_continent[df_continent['Continent'] == selected_continent]  

            # if not selected_continent or selected_continent=="":
            #     st.write('Please select a continent') 
            
            # else:
            #     with st.expander('Click to expand', expanded=True):
            #         st.subheader(f"{selected_continent} ({publications_count_con} sources)")
            #         types = st.multiselect('Publication type', df_continent['Publication type'].unique(),df_continent['Publication type'].unique(), key='original_5')
            #         df_continent = df_continent[df_continent['Publication type'].isin(types)]
            #         df_continent = df_continent.reset_index(drop=True)
            #         df_download_continent = df_continent[['Publication type','Title','FirstName2','Abstract','Date published','Publisher','Journal','Link to publication','Zotero link']]
            #         df_download_continent = df_download_continent.reset_index(drop=True)
            #         def convert_df(df_download_continent):
            #             return df_download_continent.to_csv(index=False).encode('utf-8-sig')
            #         csv = convert_df(df_download_continent)
            #         today = datetime.date.today().isoformat()
            #         a = f'{selected_continent}_{today}'
            #         st.download_button('💾 Download items', csv, (a+'.csv'), mime="text/csv", key='download-csv-6')

            #         publications_by_type_continent = df_continent['Publication type'].value_counts()
            #         num_items_collections_continent = len(df_continent)
            #         breakdown_string_continent = ', '.join([f"{key}: {value}" for key, value in publications_by_type_continent.items()])                    

            #         articles_list = []  # Store articles in a list
            #         st.write(f"**{num_items_collections_continent}** sources found ({breakdown_string_continent})")
            #         true_count = df_continent[df_continent['Publication type']=='Journal article']['OA status'].sum()
            #         total_count = len(df_continent[df_continent['Publication type']=='Journal article'])
            #         if total_count == 0:
            #             oa_ratio = 0.0
            #         else:
            #             oa_ratio = true_count / total_count * 100

            #         citation_count = df_continent['Citation'].sum()
            #         st.write(f'Number of citations: **{int(citation_count)}**, Open access coverage (journal articles only): **{int(oa_ratio)}%**')
                
            #         for index, row in df_continent.iterrows():
            #             formatted_entry = format_entry(row)  # Assuming format_entry() is a function formatting each row
            #             articles_list.append(formatted_entry)        
                    
            #         for index, row in df_continent.iterrows():
            #             publication_type = row['Publication type']
            #             title = row['Title']
            #             authors = row['FirstName2']
            #             date_published = row['Date published']
            #             link_to_publication = row['Link to publication']
            #             zotero_link = row['Zotero link']

            #             if publication_type == 'Journal article':
            #                 published_by_or_in = 'Published in'
            #                 published_source = str(row['Journal']) if pd.notnull(row['Journal']) else ''
            #             elif publication_type == 'Book':
            #                 published_by_or_in = 'Published by'
            #                 published_source = str(row['Publisher']) if pd.notnull(row['Publisher']) else ''
            #             else:
            #                 published_by_or_in = ''
            #                 published_source = ''

            #             formatted_entry = (
            #                 '**' + str(publication_type) + '**' + ': ' +
            #                 str(title) + ' ' +
            #                 '(by ' + '*' + str(authors) + '*' + ') ' +
            #                 '(Publication date: ' + str(date_published) + ') ' +
            #                 ('(' + published_by_or_in + ': ' + '*' + str(published_source) + '*' + ') ' if published_by_or_in else '') +
            #                 '[[Publication link]](' + str(link_to_publication) + ') ' +
            #                 '[[Zotero link]](' + str(zotero_link) + ')'
            #             )
            #         sort_by = st.radio('Sort by:', ('Publication date :arrow_down:', 'Publication type',  'Citation'), key='continent')
            #         display2 = st.checkbox('Display abstracts', key='type_country_999')
            #         if sort_by == 'Publication date :arrow_down:' or df_continent['Citation'].sum() == 0:
            #             count = 1
            #             for index, row in df_continent.iterrows():
            #                 formatted_entry = format_entry(row)
            #                 st.write(f"{count}) {formatted_entry}")
            #                 count += 1
            #                 if display2:
            #                     st.caption(row['Abstract']) 
            #         elif sort_by == 'Publication type' or df_continent['Citation'].sum() == 0:
            #             df_continent = df_continent.sort_values(by=['Publication type'], ascending=True)
            #             current_type = None
            #             count_by_type = {}
            #             for index, row in df_continent.iterrows():
            #                 if row['Publication type'] != current_type:
            #                     current_type = row['Publication type']
            #                     st.subheader(current_type)
            #                     count_by_type[current_type] = 1
            #                 formatted_entry = format_entry(row)
            #                 st.write(f"{count_by_type[current_type]}) {formatted_entry}")
            #                 count_by_type[current_type] += 1
            #                 if display2:
            #                     st.caption(row['Abstract'])
            #         else:
            #             df_continent = df_continent.sort_values(by=['Citation'], ascending=False)
            #             count = 1
            #             for index, row in df_continent.iterrows():
            #                 formatted_entry = format_entry(row)
            #                 st.write(f"{count}) {formatted_entry}")
            #                 count += 1
            #                 if display2:
            #                     st.caption(row['Abstract']) 

            st.subheader('Countries overview')
            col11, col12 = st.columns([3,2])
            with col11:                
                df_countries_chart = df_countries_chart[df_countries_chart['Country'] != 'Country not known']
                country_pub_counts = df_countries_chart['Country'].value_counts().sort_values(ascending=False)
                all_countries_df = pd.DataFrame({'Country': country_pub_counts.index, 'Publications': country_pub_counts.values})
                num_countries = st.slider("Select the number of countries to display", min_value=1, max_value=len(all_countries_df), value=10)
                top_countries = all_countries_df.head(num_countries).sort_values(by='Publications', ascending=True)
                fig = px.bar(top_countries, x='Publications', y='Country', orientation='h')
                fig.update_layout(title=f'Top {num_countries} Countries by Number of Publications', xaxis_title='Number of Publications', yaxis_title='Country')
                col11.plotly_chart(fig, use_container_width=True)

            with col12:
                df_continent_chart = df_continent_chart[df_continent_chart['Continent'] != 'Unknown']
                country_pub_counts = df_continent_chart['Continent'].value_counts().sort_values(ascending=False)
                top_10_countries = country_pub_counts.head(10).sort_values(ascending=True)
                top_10_df = pd.DataFrame({'Continent': top_10_countries.index, 'Publications': top_10_countries.values})
                fig = px.pie(top_10_df, values='Publications', names='Continent', title='Number of Publications by Continent')
                fig.update_layout(title='Number of Publications by continent', xaxis_title='Number of Publications', yaxis_title='Continent')
                col12.plotly_chart(fig, use_container_width = True)

            def compute_cumulative_graph(df, num_countries):
                df['Date published'] = (
                    df['Date published']
                    .str.strip()
                    .apply(lambda x: pd.to_datetime(x, utc=True, errors='coerce').tz_convert('Europe/London'))
                )
                df['Date year'] = df['Date published'].dt.strftime('%Y')
                collection_counts = df.groupby(['Date year', 'Country']).size().unstack().fillna(0)
                collection_counts = collection_counts.reset_index()
                collection_counts.iloc[:, 1:] = collection_counts.iloc[:, 1:].cumsum()

                # Select only the top countries based on the slider value
                selected_countries = top_countries['Country'].tolist()
                selected_countries = [country for country in selected_countries if country in collection_counts.columns] 

                # Check if there are still countries to display
                if not selected_countries:
                    st.warning("No data available for the selected countries.")
                else:
                    selected_columns = ['Date year'] + selected_countries
                    cumulative_selected_countries = collection_counts[selected_columns]

                    # Display the cumulative sum of publications per country
                    fig_cumulative_countries = px.line(cumulative_selected_countries, x='Date year', y=cumulative_selected_countries.columns[1:],
                                                        markers=True, line_shape='linear', labels={'value': 'Cumulative Count'},
                                                        title=f'Cumulative Publications per Country Over Years (Top {num_countries} Countries)')

                    # Reverse the legend order
                    fig_cumulative_countries.update_layout(legend_traceorder='reversed')

                return fig_cumulative_countries

            # Display the cumulative line graph based on the selected number of countries
            fig_cumulative_countries = compute_cumulative_graph(df_countries_chart, num_countries)
            st.plotly_chart(fig_cumulative_countries, use_container_width=True)

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
                event_info = evens_conferences()
                for info in event_info:
                    st.write(info)

    with tab2:
        if not selected_country=='':
            on = st.toggle("Display dashboard")
            if on:
                st.header('Dashboard')
                if not selected_country=='All Countries':
                    df_collections = df_countries.copy()

                if df_collections['Title'].any() in ("", [], None, 0, False):
                    all = st.checkbox('Show all types')
                    if all:
                        df=df_collections.copy()
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

                    # TEMPORARY SOLUTION FOR COLUMN NAME CHANGE ERROR
                    df_plot.columns = ['Publication type', 'Count']
                    # TEMP SOLUTION ENDS

                    plot= df_plot
                    # st.bar_chart(plot.sort_values(ascending=False), height=600, width=600, use_container_width=True)

                    fig = px.pie(plot, values='Count', names='Publication type')
                    fig.update_layout(title=f'Publications: {collection_name} ({selected_country})')
                    col1.plotly_chart(fig, use_container_width = True)

                with col2:
                    fig = px.bar(df_plot, x='Publication type', y='Count', color='Publication type')
                    fig.update_layout(
                        autosize=False,
                        width=400,
                        height=400,)
                    fig.update_layout(title=f'Publications: {collection_name} ({selected_country})')
                    col2.plotly_chart(fig, use_container_width = True)

                df_collections['Date published'] = pd.to_datetime(df_collections['Date published'],utc=True, errors='coerce').dt.tz_convert('Europe/London')
                df_collections['Date year'] = df_collections['Date published'].dt.strftime('%Y')
                df_collections['Date year'] = df_collections['Date year'].fillna('No date')
                df_year=df_collections['Date year'].value_counts()
                df_year=df_year.reset_index()

                col1, col2 = st.columns(2)
                with col1:
                    df_year=df_year.rename(columns={'index':'Publication year','Date year':'Count'})
                    # TEMPORARY SOLUTION FOR COLUMN NAME CHANGE ERROR
                    df_year.columns = ['Publication year', 'Count']
                    # TEMP SOLUTION ENDS
                    df_year.drop(df_year[df_year['Publication year']== 'No date'].index, inplace = True)
                    df_year=df_year.sort_values(by='Publication year', ascending=True)
                    fig = px.bar(df_year, x='Publication year', y='Count')
                    fig.update_xaxes(tickangle=-70)
                    fig.update_layout(
                        autosize=False,
                        width=400,
                        height=500,)
                    fig.update_layout(title=f'Publications: {collection_name} ({selected_country})')
                    col1.plotly_chart(fig, use_container_width = True)

                with col2:
                    df_collections['Author_name'] = df_collections['FirstName2'].apply(lambda x: x.split(', ') if isinstance(x, str) and x else x)
                    df_collections = df_collections.explode('Author_name')
                    df_collections.reset_index(drop=True, inplace=True)
                    df_collections = df_collections.loc[df_collections['Collection_Name']==collection_name]
                    df_collections['Author_name'] = df_collections['Author_name'].map(name_replacements).fillna(df_collections['Author_name'])
                    max_authors = len(df_collections['Author_name'].unique())
                    num_authors = st.slider('Select number of authors to display:', 1, min(50, max_authors), 20)
                    
                    # Adding a multiselect widget for publication types
                    # selected_types = st.multiselect('Select publication types:', df_collections['Publication type'].unique(), default=df_collections['Publication type'].unique())
                    
                    # Filtering data based on selected publication types
                    filtered_authors = df_collections[df_collections['Publication type'].isin(types)]
                    
                    if len(types) == 0:
                        st.write('No results to display')
                    else:
                        publications_by_author = filtered_authors['Author_name'].value_counts().head(num_authors)
                        fig = px.bar(publications_by_author, x=publications_by_author.index, y=publications_by_author.values)
                        fig.update_layout(
                            title=f'Top {num_authors} Authors by Publication Count ({collection_name} - {selected_country})',
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
                    # TEMPORARY SOLUTION FOR COLUMN NAME CHANGE ERROR
                    df_publisher.columns = ['Publisher', 'Count']
                    # TEMP SOLUTION ENDS
                    df_publisher = df_publisher.sort_values(['Count'], ascending=[False])
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
                                fig.update_layout(title=f'Top  {str(number)} + publishers (in log scale) ({collection_name} - {selected_country})')
                                col1.plotly_chart(fig, use_container_width = True)
                            else:
                                fig = px.bar(df_publisher, x='Publisher', y='Count', color='Publisher', log_y=True)
                                fig.update_layout(
                                    autosize=False,
                                    width=1200,
                                    height=700,
                                    showlegend=True)
                                fig.update_xaxes(tickangle=-70)
                                fig.update_layout(title=f'Top  {str(number)} + publishers (in log scale) ({collection_name} - {selected_country})')
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
                                fig.update_layout(title=f'Top  {str(number)} + publishers ({collection_name} - {selected_country})')
                                col1.plotly_chart(fig, use_container_width = True)
                            else:
                                fig = px.bar(df_publisher, x='Publisher', y='Count', color='Publisher', log_y=False)
                                fig.update_layout(
                                    autosize=False,
                                    width=1200,
                                    height=700,
                                    showlegend=True)
                                fig.update_xaxes(tickangle=-70)
                                fig.update_layout(title=f'Top  {str(number)} + publishers ({collection_name} - {selected_country})')
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
                    # TEMPORARY SOLUTION FOR COLUMN NAME CHANGE ERROR
                    df_journal.columns = ['Journal', 'Count']
                    # TEMP SOLUTION ENDS
                    df_journal = df_journal.sort_values(['Count'], ascending=[False])
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
                                fig.update_layout(title=f'Top {str(number2)} journals that publish intelligence articles (in log scale) ({collection_name} - {selected_country})')
                                col2.plotly_chart(fig, use_container_width = True)
                            else:
                                fig = px.bar(df_journal, x='Journal', y='Count', color='Journal', log_y=True)
                                fig.update_layout(
                                    autosize=False,
                                    width=1200,
                                    height=700,
                                    showlegend=True)
                                fig.update_xaxes(tickangle=-70)
                                fig.update_layout(title=f'Top {str(number2)} journals that publish intelligence articles (in log scale) ({collection_name} - {selected_country})')
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
                                fig.update_layout(title=f'Top {str(number2)} journals that publish intelligence articles ({collection_name} - {selected_country})')
                                col2.plotly_chart(fig, use_container_width = True)
                            else:
                                fig = px.bar(df_journal, x='Journal', y='Count', color='Journal', log_y=False)
                                fig.update_layout(
                                    autosize=False,
                                    width=1200,
                                    height=700,
                                    showlegend=True)
                                fig.update_xaxes(tickangle=-70)
                                fig.update_layout(title=f'Top  {str(number2)} journals that publish intelligence articles ({collection_name} - {selected_country})')
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

                SW = ['york', 'intelligence', 'security', 'pp', 'war','world', 'article', 'twitter', 'part',
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
                    plt.title(f'Top words in title (collection: {collection_name} - {selected_country}')
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
                    plt.title(f'Top words in abstract (collection: {collection_name} - {selected_country}')
                    plt.imshow(wordcloud)
                    plt.axis("off")
                    plt.show()
                    st.set_option('deprecation.showPyplotGlobalUse', False)
                    st.pyplot() 
            else:
                st.info('Toggle to see the dashboard!')
        else:
            st.warning('Select a country to display dashboard!')
    components.html(
    """
    <a rel="license" href="http://creativecommons.org/licenses/by/4.0/"><img alt="Creative Commons Licence" style="border-width:0" 
    src="https://i.creativecommons.org/l/by/4.0/80x15.png" /></a><br />
    © 2024 Yusuf Ozkan. All rights reserved. This website is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">Creative Commons Attribution 4.0 International License</a>.
    """
    )