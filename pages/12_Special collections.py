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
import plotly.express as px

st.set_page_config(layout = "wide", 
                    page_title='Intelligence bibliography',
                    page_icon="https://images.pexels.com/photos/315918/pexels-photo-315918.png",
                    initial_sidebar_state="auto") 

st.title("Special collections")

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
df_collections=df_collections[df_collections['Name'].str.contains("98.")]
df_collections = df_collections.iloc[2: , :]

# clist = df_collections['Name'].unique()

col1, col2, col3 = st.columns([1.4,4,1.6])

with col1:
    radio = st.radio('Select a collection', df_collections['Name'])
    st.markdown('''[Visuals](#visuals)''', unsafe_allow_html=True)
    
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
    columns3=['Title','Publication type', 'Link to publication', 'Abstract', 'Zotero link', 'Date published', 'FirstName2']

    for item in items:
        data3.append((
            item['data']['title'], 
            item['data']['itemType'], 
            item['data']['url'], 
            item['data']['abstractNote'], 
            item['links']['alternate']['href'],
            item['data'].get('date'),
            item['data']['creators']
            )) 
    pd.set_option('display.max_colwidth', None)

    df = pd.DataFrame(data3, columns=columns3)

    df['Date published'] = pd.to_datetime(df['Date published'], errors='coerce')
    df['Date published'] = df['Date published'].map(lambda x: x.datetime('%d/%m/%Y') if x else 'No date')
    df

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

    st.markdown('#### Collection theme: ' + collection_name)
    st.caption('This collection has ' + str(count_collection) + ' items (this number may include reviews attached to sources).') # count_collection

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
            "[[Zotero link]]" +'('+ df['Zotero link'] + ')'
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
                    "[[Zotero link]]" +'('+ df['Zotero link'] + ')'
                    )
    row_nu_1= len(df.index)
    # if row_nu_1<15:
    #     row_nu_1=row_nu_1
    # else:
    #     row_nu_1=15

    df['First author'] = df['firstName'] + ' ' + df['lastName']
    df_download = df[['Title', 'Publication type', 'First author', 'Link to publication', 'Zotero link']]

    def convert_df(df):
        return df.to_csv(index=False).encode('utf-8-sig') # not utf-8 because of the weird character,  Â cp1252
    today = datetime.date.today().isoformat()
    csv = convert_df(df_download)
    # csv = df_download
    # # st.caption(collection_name)
    st.download_button('Download the collection', csv, collection_name+ '-'+today +'.csv', mime="text/csv", key='download-csv')

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

st.header('Visuals')

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



components.html(
"""
<a rel="license" href="http://creativecommons.org/licenses/by/4.0/"><img alt="Creative Commons Licence" style="border-width:0" 
src="https://i.creativecommons.org/l/by/4.0/80x15.png" /></a><br />
© 2022 All rights reserved. This website is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">Creative Commons Attribution 4.0 International License</a>.
"""
)
