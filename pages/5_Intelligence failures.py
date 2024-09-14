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
# from gsheetsdb import connect
from streamlit_gsheets import GSheetsConnection
import datetime as dt     
import random
from authors_dict import df_authors, name_replacements
from sidebar_content import sidebar_content, set_page_config
from format_entry import format_entry
from copyright import display_custom_license
from events import evens_conferences
from st_keyup import st_keyup


set_page_config()

st.title("Intelligence failures", anchor=False)

with st.spinner('Retrieving data & updating dashboard...'):

    # # Connecting Zotero with API
    # library_id = '2514686' # intel 2514686
    # library_type = 'group'
    # api_key = '' # api_key is only needed for private groups and libraries

    sidebar_content()

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
    # df_collections = df_collections[~df_collections['Collection_Name'].str.contains('01.98')]
    # df_collections = df_collections[df_collections['Collection_Name'] != '01 Intelligence history']

    df_collections = df_collections.sort_values(by='Collection_Name')
    df_collections=df_collections[df_collections['Collection_Name'].str.contains("05.")]

    def remove_numbers(name):
        return re.sub(r'^\d+(\.\d+)*\s*', '', name)

    df_collections['Collection_Name'] = df_collections['Collection_Name'].apply(remove_numbers)

    collection_mapping = df_collections.drop_duplicates('Collection_Name').set_index('Collection_Name')['Collection_Key'].to_dict()
    reverse_collection_mapping = {v: k for k, v in collection_mapping.items()}

    st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)

    container = st.container()

    unique_collections = list(df_collections['Collection_Name'].unique())

    def update_params():
        st.query_params.from_dict({'collection_id': collection_mapping[st.session_state.qp]})

    query_params = st.query_params
    ix = 0

    if 'collection_id' in query_params:
        try:
            # Get the collection name using the key from query_params
            collection_name_from_key = reverse_collection_mapping[query_params['collection_id']]
            ix = unique_collections.index(collection_name_from_key)
        except (ValueError, KeyError):
            pass

    radio = container.radio('Select a collection', unique_collections, index=ix, key="qp", on_change=update_params)
    query_params = st.query_params.to_dict()

    collection_name = radio
    collection_key = collection_mapping[collection_name]

    df_collections = df_collections.loc[df_collections['Collection_Name']==collection_name]
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

    st.markdown('#### Collection theme: ' + collection_name)

    # name = st.text_input("Enter keywords to search in title", key='name', placeholder='Search keyword(s)')#, debounce=250, key='name')
    name = st_keyup("Enter keywords to search in title", key='name', placeholder='Search keyword(s)', debounce=500)#, debounce=250, key='name')
    if name:
        df_collections = df_collections[df_collections.Title.str.lower().str.contains(name.lower(), na=False)]

    col1, col2, col3 = st.columns([1,2,4])
    with col1:
        container_metric = st.container()
    with col2:
        with st.popover("More metrics"):
            container_citation = st.container()
            container_citation_average = st.container()
            container_oa = st.container()
            container_type = st.container()
            container_author_no = st.container()
            container_author_pub_ratio = st.container()
            container_publication_ratio = st.container()
    with col3:
        with st.popover("Filters and more"):
            st.write(f"View the collection in [Zotero]({collection_link})")
            col112, col113 = st.columns(2)
            with col112:
                display2 = st.checkbox('Display abstracts')
            with col113:
                only_citation = st.checkbox('Show cited items only')
                if only_citation:
                    df_collections = df_collections[(df_collections['Citation'].notna()) & (df_collections['Citation'] != 0)]
            view = st.radio('View as:', ('Basic list', 'Table',  'Bibliography'))

            types = st.multiselect('Publication type', df_collections['Publication type'].unique(),df_collections['Publication type'].unique(), key='original')
            df_collections = df_collections[df_collections['Publication type'].isin(types)]
            df_collections = df_collections.reset_index(drop=True)
            df_collections['FirstName2'] = df_collections['FirstName2'].map(name_replacements).fillna(df_collections['FirstName2'])
            df_download = df_collections[['Publication type','Title','FirstName2','Abstract','Date published','Publisher','Journal','Link to publication','Zotero link']]
            df_download = df_download.reset_index(drop=True)
            df_download['Abstract'] = df_download['Abstract'].str.replace('\n', ' ')
            def convert_df(df_download):
                return df_download.to_csv(index=False).encode('utf-8-sig')
            csv = convert_df(df_download)
            today = datetime.date.today().isoformat()
            num_items_collections = len(df_collections)
            publications_by_type = df_collections['Publication type'].value_counts()
            breakdown_string = ', '.join([f"{key}: {value}" for key, value in publications_by_type.items()])
            item_type_no = df_collections['Publication type'].nunique()
            def split_and_expand(authors):
                # Ensure the input is a string
                if isinstance(authors, str):
                    # Split by comma and strip whitespace
                    split_authors = [author.strip() for author in authors.split(',')]
                    return pd.Series(split_authors)
                else:
                    # Return the original author if it's not a string
                    return pd.Series([authors])
            if len(df_collections) == 0:
                author_pub_ratio=0.0
                author_no=0
            else:
                expanded_authors = df_collections['FirstName2'].apply(split_and_expand).stack().reset_index(level=1, drop=True)
                expanded_authors = expanded_authors.reset_index(name='Author')
                author_no = len(expanded_authors)
                author_pub_ratio = round(author_no/num_items_collections, 2)

            true_count = df_collections[df_collections['Publication type']=='Journal article']['OA status'].sum()
            total_count = len(df_collections[df_collections['Publication type']=='Journal article'])
            if total_count == 0:
                oa_ratio = 0.0
            else:
                oa_ratio = true_count / total_count * 100

            citation_count = df_collections['Citation'].sum()


            a = f'{collection_name}_{today}'
            st.download_button('💾 Download the collection', csv, (a+'.csv'), mime="text/csv", key='download-csv-4')

    container_metric.metric(label="Items found", value=num_items_collections, help=breakdown_string)
    container_citation.metric(label="Number of citations", value=int(citation_count))
    container_oa.metric(label="Open access coverage", value=f'{int(oa_ratio)}%', help='Journal articles only')
    container_type.metric(label='Number of publication types', value=int(item_type_no))
    container_author_no.metric(label='Number of authors', value=int(author_no))
    container_author_pub_ratio.metric(label='Author/publication ratio', value=author_pub_ratio, help='The average author number per publication')

    outlier_detector = (df_collections['Citation'] > 1000).any()
    if outlier_detector == True:
        outlier_count = (df_collections['Citation'] > 1000).sum()
        citation_average = df_collections[df_collections['Citation'] < 1000]
        citation_average = round(citation_average['Citation'].mean(), 2)
        citation_average_with_outliers = round(df_collections['Citation'].mean(), 2)
        container_citation_average.metric(
            label="Average citation", 
            value=citation_average, 
            help=f'**{outlier_count}** item(s) passed the threshold of 1000 citations. With the outliers, the average citation count is **{citation_average_with_outliers}**.'
            )
    citation_average = round(df_collections['Citation'].mean(), 2)
    container_citation_average.metric(label="Average citation", value=citation_average)

    df_collections['FirstName2'] = df_collections['FirstName2'].astype(str)
    df_collections['multiple_authors'] = df_collections['FirstName2'].apply(lambda x: ',' in x)
    if len(df_collections) == 0:
        collaboration_ratio=0
    else:
        multiple_authored_papers = df_collections['multiple_authors'].sum()
        collaboration_ratio = round(multiple_authored_papers / num_items_collections * 100, 1)
        container_publication_ratio.metric(label='Collaboration ratio', value=f'{(collaboration_ratio)}%', help='Ratio of multiple-authored papers')

    tab1, tab2 = st.tabs(['📑 Publications', '📊 Dashboard'])
    with tab1:
        col1, col2 = st.columns([5,1.6])
        with col1:            
            # st.write(f"**{num_items_collections}** sources found ({breakdown_string})")
            # st.write(f'Number of citations: **{int(citation_count)}**, Open access coverage (journal articles only): **{int(oa_ratio)}%**')
            # THIS WAS THE PLACE WHERE FORMAT_ENTRY WAS LOCATED
            sort_by = st.radio('Sort by:', ('Publication date :arrow_down:', 'Publication type',  'Citation', 'Date added :arrow_down:'))

            if sort_by == 'Publication date :arrow_down:': # or df_collections['Citation'].sum() == 0:
                df_collections = df_collections.sort_values(by=['Date published'], ascending=False)
                df_collections = df_collections.reset_index()

            elif sort_by == 'Publication type': # or df_collections['Citation'].sum() == 0:
                df_collections = df_collections.sort_values(by=['Publication type'], ascending=True)
                df_collections = df_collections.reset_index()

            elif sort_by =='Citation':
                df_collections = df_collections.sort_values(by=['Citation'], ascending=False)
                df_collections = df_collections.reset_index()

            else: #elif sort_by == 'Date added :arrow_down:':
                df_collections = df_collections.sort_values(by=['Date added'], ascending=False)
                df_collections = df_collections.reset_index()

            if view == 'Basic list':
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
                                
                with st.expander('**Basic list view**', expanded=True):

                    if sort_by == 'Publication date :arrow_down:': # or df_collections['Citation'].sum() == 0:
                        count = 1
                        for index, row in df_collections.iterrows():
                            formatted_entry = format_entry(row)
                            st.write(f"{count}) {formatted_entry}")
                            count += 1
                            if display2:
                                st.caption(row['Abstract']) 
                    elif sort_by == 'Publication type': # or df_collections['Citation'].sum() == 0:
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
                    elif sort_by =='Citation':
                        if df_collections['Citation'].sum() == 0:
                            count = 1
                            for index, row in df_collections.iterrows():
                                formatted_entry = format_entry(row)
                                st.write(f"{count}) {formatted_entry}")
                                count += 1
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
                    else: #elif sort_by == 'Date added :arrow_down:':
                        df_collections = df_collections.sort_values(by=['Date added'], ascending=False)
                        count = 1
                        for index, row in df_collections.iterrows():
                            formatted_entry = format_entry(row)
                            st.write(f"{count}) {formatted_entry}")
                            count += 1
                            if display2:
                                st.caption(row['Abstract']) 
            elif view == 'Table':
                df_table_view = df_collections[['Publication type','Title','Date published','FirstName2', 'Abstract','Publisher','Journal', 'Citation', 'Collection_Name','Link to publication','Zotero link']]
                df_table_view = df_table_view.rename(columns={'FirstName2':'Author(s)','Collection_Name':'Collection','Link to publication':'Publication link'})
                if sort_by == 'Publication type':
                    df_table_view = df_table_view.sort_values(by=['Publication type'], ascending=True)
                    df_table_view = df_table_view.reset_index(drop=True)
                elif sort_by == 'Citation':
                    df_table_view = df_table_view.sort_values(by=['Citation'], ascending=False)
                    df_table_view = df_table_view.reset_index(drop=True)
                with st.expander('**Table view**', expanded=True):
                    df_table_view
            else:
                with st.expander('**Bibliographic listing**', expanded=True):
                    df_collections['zotero_item_key'] = df_collections['Zotero link'].str.replace('https://www.zotero.org/groups/intelligence_bibliography/items/', '')
                    df_zotero_id = pd.read_csv('zotero_citation_format.csv')
                    df_collections = pd.merge(df_collections, df_zotero_id, on='zotero_item_key', how='left')
                    df_zotero_id = df_collections[['zotero_item_key']]

                    def display_bibliographies(df):
                        df['bibliography'] = df['bibliography'].fillna('').astype(str)
                        all_bibliographies = ""
                        for index, row in df.iterrows():
                            # Add a horizontal line between bibliographies
                            if index > 0:
                                all_bibliographies += '<p><p>'
                            
                            # Display bibliography
                            all_bibliographies += row['bibliography']

                        st.markdown(all_bibliographies, unsafe_allow_html=True)

                    # Streamlit app

                    # Display bibliographies from df_collections DataFrame
                    display_bibliographies(df_collections)

#UNTIL HERE
        with col2:
            @st.experimental_fragment
            def events():
                with st.expander('Collections', expanded=True):
                    if st.button(
                        'Home',
                    ):
                        st.switch_page('Home_page.py')
                    if st.button(
                        'Intelligence history',
                    ):
                        st.switch_page('pages/1_Intelligence history.py')
                    if st.button(
                        'Intelligence studies',
                    ):
                        st.switch_page('pages/2_Intelligence studies.py')
                    if st.button(
                        'Intelligence analysis',
                    ):
                        st.switch_page('pages/3_Intelligence analysis.py')
                    if st.button(
                        'Intelligence organisations',
                    ):
                        st.switch_page('pages/4_Intelligence organisations.py')
                    if st.button(
                        'Intelligence failures',
                    ):
                        st.switch_page('pages/5_Intelligence failures.py')
                    if st.button(
                        'Intelligence oversight and ethics',
                    ):
                        st.switch_page('pages/6_Intelligence oversight and ethics.py')
                    if st.button(
                        'Intelligence collection',
                    ):
                        st.switch_page('pages/7_Intelligence collection.py')
                    if st.button(
                        'Counterintelligence',
                    ):
                        st.switch_page('pages/8_Counterintelligence.py')
                    if st.button(
                        'Covert action',
                    ):
                        st.switch_page('pages/9_Covert action.py')
                    if st.button(
                        'Intelligence and cybersphere',
                    ):
                        st.switch_page('pages/10_Intelligence and cybersphere.py')
                    if st.button(
                        'Global intelligence',
                    ):
                        st.switch_page('pages/11_Global intelligence.py')
                    if st.button(
                        'Special collections',
                    ):
                        st.switch_page('pages/12_Special collections.py')
            events()
                # st.caption('[Intelligence history](https://intelligence.streamlit.app/Intelligence_history)')
                # st.caption('[Intelligence studies](https://intelligence.streamlit.app/Intelligence_studies)')
                # st.caption('[Intelligence analysis](https://intelligence.streamlit.app/Intelligence_analysis)')
                # st.caption('[Intelligence organisations](https://intelligence.streamlit.app/Intelligence_organisations)')
                # st.caption('[Intelligence failures](https://intelligence.streamlit.app/Intelligence_failures)')
                # st.caption('[Intelligence oversight and ethics](https://intelligence.streamlit.app/Intelligence_oversight_and_ethics)')
                # st.caption('[Intelligence collection](https://intelligence.streamlit.app/Intelligence_collection)')
                # st.caption('[Counterintelligence](https://intelligence.streamlit.app/Counterintelligence)')
                # st.caption('[Covert action](https://intelligence.streamlit.app/Covert_action)')
                # st.caption('[Intelligence and cybersphere](https://intelligence.streamlit.app/Intelligence_and_cybersphere)')
                # st.caption('[Global intelligence](https://intelligence.streamlit.app/Global_intelligence)')
                # st.caption('[AI and intelligence](https://intelligence.streamlit.app/AI_and_intelligence)')
                # st.caption('[Special collections](https://intelligence.streamlit.app/Special_collections)')
            with st.expander('Events', expanded=True):
                # Create a connection object.
                event_info = evens_conferences()
                for info in event_info:
                    st.write(info)

    with tab2:
        st.header('Dashboard')
        on = st.toggle('Display dashboard')
        if on:
            if df_collections['Title'].any() in ("", [], None, 0, False):
                all = st.checkbox('Show all types')
                if all:
                    df=df_collections.copy()
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
                fig.update_layout(title={'text':'Publications: '+collection_name})
                col1.plotly_chart(fig, use_container_width = True)

            with col2:
                fig = px.bar(df_plot, x='Publication type', y='Count', color='Publication type')
                fig.update_layout(
                    autosize=False,
                    width=400,
                    height=400,)
                fig.update_layout(title={'text':'Publications: '+collection_name})
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
                fig.update_layout(title={'text':'Publications by year: '+collection_name})
                col1.plotly_chart(fig, use_container_width = True)

            with col2:
                df_collections['Author_name'] = df_collections['FirstName2'].apply(lambda x: x.split(', ') if isinstance(x, str) and x else x)
                df_collections = df_collections.explode('Author_name')
                df_collections.reset_index(drop=True, inplace=True)
                df_collections = df_collections.loc[df_collections['Collection_Name']==collection_name]
                df_collections['Author_name'] = df_collections['Author_name'].map(name_replacements).fillna(df_collections['Author_name'])
                max_authors = len(df_collections['Author_name'].unique())
                num_authors = st.slider('Select number of authors to display:', 1, min(50, max_authors), 20)

                
                # Filtering data based on selected publication types
                filtered_authors = df_collections[df_collections['Publication type'].isin(types)]
                
                if len(df_collections['Author_name'].unique()) == 0:
                    st.write('No results to display')
                else:
                    publications_by_author = filtered_authors['Author_name'].value_counts().head(num_authors)
                    fig = px.bar(publications_by_author, x=publications_by_author.index, y=publications_by_author.values)
                    fig.update_layout(
                        title=f'Top {num_authors} Authors by Publication Count ({collection_name})',
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
                            fig.update_layout(title={'text':'Top ' + str(number) + ' publishers (in log scale)'})
                            col1.plotly_chart(fig, use_container_width = True)
                        else:
                            fig = px.bar(df_publisher, x='Publisher', y='Count', color='Publisher', log_y=True)
                            fig.update_layout(
                                autosize=False,
                                width=1200,
                                height=700,
                                showlegend=True)
                            fig.update_xaxes(tickangle=-70)
                            fig.update_layout(title={'text':'Top ' + str(number) + ' publishers (in log scale)'})
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
                            fig.update_layout(title={'text':'Top ' + str(number) + ' publishers'})
                            col1.plotly_chart(fig, use_container_width = True)
                        else:
                            fig = px.bar(df_publisher, x='Publisher', y='Count', color='Publisher', log_y=False)
                            fig.update_layout(
                                autosize=False,
                                width=1200,
                                height=700,
                                showlegend=True)
                            fig.update_xaxes(tickangle=-70)
                            fig.update_layout(title={'text':'Top ' + str(number) + ' publishers'})
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
                            fig.update_layout(title={'text':'Top ' + str(number2) + ' journals that publish intelligence articles (in log scale)'})
                            col2.plotly_chart(fig, use_container_width = True)
                        else:
                            fig = px.bar(df_journal, x='Journal', y='Count', color='Journal', log_y=True)
                            fig.update_layout(
                                autosize=False,
                                width=1200,
                                height=700,
                                showlegend=True)
                            fig.update_xaxes(tickangle=-70)
                            fig.update_layout(title={'text':'Top ' + str(number2) + ' journals that publish intelligence articles (in log scale)'})
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
                            fig.update_layout(title={'text':'Top ' + str(number2) + ' journals that publish intelligence articles'})
                            col2.plotly_chart(fig, use_container_width = True)
                        else:
                            fig = px.bar(df_journal, x='Journal', y='Count', color='Journal', log_y=False)
                            fig.update_layout(
                                autosize=False,
                                width=1200,
                                height=700,
                                showlegend=True)
                            fig.update_xaxes(tickangle=-70)
                            fig.update_layout(title={'text':'Top ' + str(number2) + ' journals that publish intelligence articles)'})
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

            SW = ['york', 'intelligence', 'security', 'pp', 'war','world', 'article', 'twitter', 'failure',
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
                # st.set_option('deprecation.showPyplotGlobalUse', False)
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
                # st.set_option('deprecation.showPyplotGlobalUse', False)
                st.pyplot() 
        else:
            st.info('Toggle to see the dashboard!')
            
    display_custom_license()
