from pyzotero import zotero
import pandas as pd
import streamlit as st
from IPython.display import HTML
import streamlit.components.v1 as components
import numpy as np
import altair as alt
from pandas.io.json import json_normalize

st.set_page_config(layout = "wide", 
                    page_title='Intelligence bibliography',
                    page_icon="https://images.pexels.com/photos/315918/pexels-photo-315918.png",
                    initial_sidebar_state="auto") 

st.title("Intelligence history")

# Connecting Zotero with API
library_id = '2514686' # intel 2514686
library_type = 'group'
api_key = '' # api_key is only needed for private groups and libraries


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

zot = zotero.Zotero(library_id, library_type)

collections = zot.collections()
data2=[]
columns2 = ['Key','Name', 'Link']
for item in collections:
    data2.append((item['data']['key'], item['data']['name'], item['links']['alternate']['href']))

pd.set_option('display.max_colwidth', None)
df_collections = pd.DataFrame(data2, columns=columns2)

df_collections = df_collections.sort_values(by='Name')
df_collections=df_collections[df_collections['Name'].str.contains("01.")]
df_collections = df_collections.iloc[1: , :]

# clist = df_collections['Name'].unique()

col1, col2, col3 = st.columns([1.5,4,1.6])

with col1:
    radio = st.radio('Select a collection', df_collections['Name'])
    
    # collection_name = st.selectbox('Select a collection:', clist)
    collection_name = radio
    collection_code = df_collections.loc[df_collections['Name']==collection_name, 'Key'].values[0]

    df_collections=df_collections['Name'].reset_index()
    pd.set_option('display.max_colwidth', None)

with col2:
# Collection items

    count_collection = zot.num_collectionitems(collection_code)

    items = zot.everything(zot.collection_items_top(collection_code))

    data3=[]
    columns3=['Title','Publication type', 'Link to publication', 'Abstract', 'Zotero link', 'FirstName2']

    for item in items:
        data3.append((
            item['data']['title'], 
            item['data']['itemType'], 
            item['data']['url'], 
            item['data']['abstractNote'], 
            item['links']['alternate']['href'],
            item['data']['creators']
            )) 
    pd.set_option('display.max_colwidth', None)

    df = pd.DataFrame(data3, columns=columns3)

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

    df_items = ('**'+ df['Publication type']+ '**'+ ': ' +
                df['Title'] + ' '+ 
                # ' (by ' + '*' + df['firstName'] + '*'+ ' ' + '*' + df['lastName'] + '*' + ') ' + # IT CANNOT READ THE NAN VALUES
                "[[Publication link]]" +'('+ df['Link to publication'] + ')' +'  '+
                "[[Zotero link]]" +'('+ df['Zotero link'] + ')'
                )
     
    row_nu_1= len(df.index)
    if row_nu_1<15:
        row_nu_1=row_nu_1
    else:
        row_nu_1=15
    df_download = df.drop(df.columns[['Abstract', 'FirstName2']], axis=1, inplace=True)
    df_download
    
    st.markdown('#### Collection theme: ' + collection_name)
    st.caption('This collection has ' + str(count_collection) + ' items (this number may include reviews attached to sources).')
    with st.expander("Expand to see the list", expanded=True):
        st.write('To see the collection in Zotero click [here](https://www.zotero.org/groups/2514686/intelligence_bibliography/collections/' + collection_code + ')')
        # display2 = st.checkbox('Display abstracts')
        for i in range(row_nu_1):
            st.write(''+str(i+1)+') ' +df_items.iloc[i])
            df_items.fillna("nan") 
            # if display2:
            #     st.caption(df['Abstract'].iloc[i])

with col3:
    with st.expander("Collections in Zotero library", expanded=False):
        bbb = zot.collections()
        data3=[]
        columns3 = ['Key','Name', 'Number', 'Link']
        for item in bbb:
            data3.append((item['data']['key'], item['data']['name'], item['meta']['numItems'], item['links']['alternate']['href']))
        pd.set_option('display.max_colwidth', None)
        df_collections_2 = pd.DataFrame(data3, columns=columns3)
        row_nu_collections = len(df_collections_2.index)
        
        for i in range(row_nu_collections):
            st.caption('[' + df_collections_2.sort_values(by='Name')['Name'].iloc[i]+ ']'+ '('+ df_collections_2.sort_values(by='Name')['Link'].iloc[i] + ')' + 
            ' [' + str(df_collections_2.sort_values(by='Name')['Number'].iloc[i]) + ' items]'
            )