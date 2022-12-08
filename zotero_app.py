from pyzotero import zotero
import pandas as pd
import streamlit as st
from IPython.display import HTML
import streamlit.components.v1 as components
import numpy as np
import altair as alt
# from bokeh.models.widgets import Button
# from bokeh.models import CustomJS
# from streamlit_bokeh_events import streamlit_bokeh_events

# Connecting Zotero with API
library_id = '2514686'
library_type = 'group'
api_key = '' # api_key is only needed for private groups and libraries

# Bringing recently changed items
zot = zotero.Zotero(library_id, library_type)
items = zot.top(limit=10)

data=[]
columns = ['Title','Publication type', 'Link to publication', 'Abstract', 'Zotero link', 'Date added', 'Col key', 'FirstName']

for item in items:
    data.append((item['data']['title'], 
    item['data']['itemType'], 
    item['data']['url'], 
    item['data']['abstractNote'], 
    item['links']['alternate']['href'], 
    item['data']['dateAdded'], 
    item['data']['collections'],
    item['data']['creators']
    ))

st.set_page_config(layout = "wide", 
                    page_title='Intelligence bibliography',
                    page_icon="https://images.pexels.com/photos/315918/pexels-photo-315918.png",
                    initial_sidebar_state="auto") 
pd.set_option('display.max_colwidth', None)
df = pd.DataFrame(data, columns=columns)
df['FirstName'] = pd.DataFrame(df['FirstName'].tolist())
split_df= pd.DataFrame(df['Col key'].tolist())
df = pd.concat([df, split_df], axis=1)
df['FirstName'] = str(df['FirstName']).fillna(0)

df

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


# Bringing collections
bbb = zot.collections()
data3=[]
columns3 = ['Key','Name', 'Number', 'Link']
for item in bbb:
    data3.append((item['data']['key'], item['data']['name'], item['meta']['numItems'], item['links']['alternate']['href']))
pd.set_option('display.max_colwidth', None)
df_collections_2 = pd.DataFrame(data3, columns=columns3)

collections = zot.collections()
data2=[]
columns2 = ['Key','Name', 'Link']
for item in collections:
    data2.append((item['data']['key'], item['data']['name'], item['links']['alternate']['href']))

pd.set_option('display.max_colwidth', None)
df_collections = pd.DataFrame(data2, columns=columns2)

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

st.title("Intelligence bibliography")
# st.header("[Zotero group library](https://www.zotero.org/groups/2514686/intelligence_bibliography/library)")

count = zot.count_items()
st.write('There are '+  '**'+str(count)+ '**' + ' items in the Zotero group library. To see the full library, go to the [Intelligence bibliography Zotero group library](https://www.zotero.org/groups/2514686/intelligence_bibliography/items).')
df['Date added'] = pd.to_datetime(df['Date added'], errors='coerce')
df['Date added'] = df['Date added'].dt.strftime('%d/%m/%Y')
st.write('The library last updated on ' + '**'+ df.loc[0]['Date added']+'**')

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

# Recently added items

col1, col2 = st.columns([4,1]) 
with col1:
    st.header('Recently added items: ')
    with st.expander('Click to hide the list', expanded=True):
        display = st.checkbox('Display theme and abstract')

        df_last = ('**'+ df['Publication type']+ '**'+ ': ' +  df['Title'] + ' '+ 
        "[[Publication link]]" +'('+ df['Link to publication'] + ')' +'  '+ 
        "[[Zotero link]]" +'('+ df['Zotero link'] + ')' + ' (Added on: ' + df['Date added']+')'
        )
        row_nu_1= len(df_last.index)
        for i in range(row_nu_1):
            st.write(''+str(i+1)+') ' +df_last.iloc[i])
            if display:
                if 0 in df:
                    st.caption('Theme(s):  \n ' + '['+df['Name_x'].iloc[i]+']' +'('+ df['Link_x'].iloc[i] + ')') 
                    if 1 in df:
                        st.caption('['+df['Name_y'].iloc[i]+']' +'('+ df['Link_y'].iloc[i] + ')')
                        if 2 in df:
                            st.caption('['+df['Name'].iloc[i]+']' +'('+ df['Link'].iloc[i] + ')')
                else:
                    st.caption('No theme to display!')
                st.caption('Abstract:'+'\n '+ df['Abstract'].iloc[i])
# Collection list

    st.header('Items by collection: ')
    clist = df_collections['Name'].unique()
    collection_name = st.selectbox('Select a collection:', clist)
    collection_code = df_collections.loc[df_collections['Name']==collection_name, 'Key'].values[0]

    df_collections=df_collections['Name'].reset_index()
    pd.set_option('display.max_colwidth', None)

    # Collection items

    count_collection = zot.num_collectionitems(collection_code)

    items = zot.everything(zot.collection_items_top(collection_code))

    data3=[]
    columns3=['Title','Publication type', 'Link to publication', 'Abstract', 'Zotero link']

    for item in items:
        data3.append((item['data']['title'], item['data']['itemType'], item['data']['url'], item['data']['abstractNote'], item['links']['alternate']['href'])) 
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

    df_items = '**'+ df['Publication type']+ '**'+ ': ' +  df['Title'] + ' '+ "[[Publication link]]" +'('+ df['Link to publication'] + ')' +'  '+ "[[Zotero link]]" +'('+ df['Zotero link'] + ')'

    row_nu_1= len(df_items.index)
    if row_nu_1<15:
        row_nu_1=row_nu_1
    else:
        row_nu_1=15

    st.markdown('#### Collection theme: ' + collection_name)
    st.caption('This collection has ' + str(count_collection) + ' items.')
    with st.expander("Expand to see the list", expanded=False):
        st.write('This list shows the last 15 added items. To see the full collection list click [here](https://www.zotero.org/groups/2514686/intelligence_bibliography/collections/' + collection_code + ')')
        # display2 = st.checkbox('Display abstracts')
        for i in range(row_nu_1):
            st.write(''+str(i+1)+') ' +df_items.iloc[i])
            df_items.fillna("nan") 
            # if display2:
            #     st.caption(df['Abstract'].iloc[i])

with col2:
    with st.expander("Collections in Zotero library", expanded=False):
        row_nu_collections = len(df_collections_2.index)
        
        for i in range(row_nu_collections):
            st.caption('[' + df_collections_2.sort_values(by='Name')['Name'].iloc[i]+ ']'+ '('+ df_collections_2.sort_values(by='Name')['Link'].iloc[i] + ')' + 
            ' [' + str(df_collections_2.sort_values(by='Name')['Number'].iloc[i]) + ' items]'
            )

    # Zotero library collections

st.header('Top 10 collections in the library: ')
df_collections_2.set_index('Name', inplace=True)
df_collections_2 = df_collections_2.sort_values(['Number'], ascending=[False])
plot= df_collections_2.head(10)

st.bar_chart(plot['Number'].sort_values(), height=600, width=600, use_container_width=True)


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


components.html(
"""
<a rel="license" href="http://creativecommons.org/licenses/by/4.0/"><img alt="Creative Commons Licence" style="border-width:0" 
src="https://i.creativecommons.org/l/by/4.0/80x15.png" /></a><br />
© 2022 All rights reserved. This website is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">Creative Commons Attribution 4.0 International License</a>.
"""
)