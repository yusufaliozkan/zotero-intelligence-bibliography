# Libraries
from pyzotero import zotero
import pandas as pd
import streamlit as st
from IPython.display import HTML
import streamlit.components.v1 as components
import numpy as np
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
columns = ['Title','Publication type', 'Link to publication', 'Abstract', 'Zotero link', 'Date added', 'Col key']

for item in items:
    data.append((item['data']['title'], 
    item['data']['itemType'], 
    item['data']['url'], 
    item['data']['abstractNote'], 
    item['links']['alternate']['href'], 
    item['data']['dateAdded'], 
    item['data']['collections']
    ))

st.set_page_config(layout = "wide", 
                    page_title='Intelligence bibliography',
                    page_icon="https://images.pexels.com/photos/315918/pexels-photo-315918.png",
                    initial_sidebar_state="auto") 
pd.set_option('display.max_colwidth', None)
df = pd.DataFrame(data, columns=columns)
split_df= pd.DataFrame(df['Col key'].tolist(), columns=['col1', 'col2', 'col3']) # https://datascienceparichay.com/article/split-pandas-column-of-lists-into-multiple-columns/ 
df = pd.concat([df, split_df], axis=1)

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
columns2 = ['Key','Name', 'Number', 'Link']
for item in collections:
    data2.append((item['data']['key'], item['data']['name'], item['meta']['numItems'], item['links']['alternate']['href']))

pd.set_option('display.max_colwidth', None)
df_collections = pd.DataFrame(data2, columns=columns2)

df_collections = df_collections.sort_values(by='Name')

df['Col1Name'] = df['col1'].map(df_collections['Name'])

merged_df = pd.merge(
    left=df,
    right=df_collections,
    left_on='col1',
    right_on='Key',
    how='left'
)
merged_df2 = pd.merge(
    left=merged_df,
    right=df_collections,
    left_on='col2',
    right_on='Key',
    how='left'
)
merged_df3 = pd.merge(
    left=merged_df2,
    right=df_collections,
    left_on='col3',
    right_on='Key',
    how='left'
)
df = merged_df3.copy()
df = df.fillna('')
df= df.drop(columns=['Number', 'Number_x', 'Number_y'])
df
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

col1, col2 = st.columns([4,2]) 
with col1:
    st.header('Recently added items: ')
    with st.expander('Click to hide the list', expanded=True):
        display = st.checkbox('Display theme/abstract')

        df_last = ('**'+ df['Publication type']+ '**'+ ': ' +  df['Title'] + ' '+ 
        "[[Publication link]]" +'('+ df['Link to publication'] + ')' +'  '+ 
        "[[Zotero link]]" +'('+ df['Zotero link'] + ')' + ' (Added on: ' + df['Date added']+')'
        )
        row_nu_1= len(df_last.index)
        for i in range(row_nu_1):
            st.write(''+str(i+1)+') ' +df_last.iloc[i])
            if display:
                st.caption('Themes: ' + '[['+df['Name_x'].iloc[i]+']]' +'('+ df['Link_x'].iloc[i] + ')' + ' ' +
                '[['+df['Name_y'].iloc[i]+']]' +'('+ df['Link_y'].iloc[i] + ')'+' ' +
                '[['+df['Name'].iloc[i]+']]' +'('+ df['Link'].iloc[i] + ')'
                )
                st.caption('Abstract:'+'\n '+ df['Abstract'].iloc[i])

# Collection list

    st.header('Items by collection: ')
    clist = df_collections['Name'].unique()
    collection_name = st.selectbox('Select a collection:', clist)
    collection_code = df_collections.loc[df_collections['Name']==collection_name, 'Key'].values[0]

    df_collections=df_collections['Name'].reset_index()
    pd.set_option('display.max_colwidth', None)

    # Collection items

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
            st.caption('[' + df_collections_2.sort_values(by='Name')['Name'].iloc[i]+ ']'+ '('+ df_collections_2.sort_values(by='Name')['Link'].iloc[i] + ')')

    # Zotero library collections

components.html(
"""
<a rel="license" href="http://creativecommons.org/licenses/by/4.0/"><img alt="Creative Commons Licence" style="border-width:0" 
src="https://i.creativecommons.org/l/by/4.0/80x15.png" /></a><br />
© 2022 All rights reserved. This website is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">Creative Commons Attribution 4.0 International License</a>.
"""
)


# Legacy

# # Libraries
# from pyzotero import zotero
# import pandas as pd
# import streamlit as st
# from IPython.display import HTML
# import streamlit.components.v1 as components
# import numpy as np
# # from bokeh.models.widgets import Button
# # from bokeh.models import CustomJS
# # from streamlit_bokeh_events import streamlit_bokeh_events

# library_id = '2514686'
# library_type = 'group'
# api_key = '' # api_key is only needed for private groups and libraries

# zot = zotero.Zotero(library_id, library_type)
# items = zot.top(limit=10)
# # items = zot.items()

# # items = zot.collection_items_top('BNPYHVD4', limit=10)
# pd.set_option('display.max_colwidth', None)

# data=[]
# columns = ['Title','Publication type', 'Link to publication', 'Abstract', 'Zotero link', 'Date added', 'Col key', 'Data']

# for item in items:
#     data.append((item['data']['title'], 
#     item['data']['itemType'], 
#     item['data']['url'], 
#     item['data']['abstractNote'], 
#     item['links']['alternate']['href'], 
#     item['data']['dateAdded'], 
#     item['data']['collections'],
#     item['data']
#     ))

# st.set_page_config(layout = "wide", 
#                     page_title='Intelligence bibliography',
#                     page_icon="https://images.pexels.com/photos/315918/pexels-photo-315918.png",
#                     initial_sidebar_state="auto") 

# df = pd.DataFrame(data, columns=columns)
# df['Col key']
# split_df= pd.DataFrame(df['Col key'].tolist()) # https://datascienceparichay.com/article/split-pandas-column-of-lists-into-multiple-columns/ 
# split_df


# # Change type name
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



# st.title("Intelligence bibliography")
# # st.header("[Zotero group library](https://www.zotero.org/groups/2514686/intelligence_bibliography/library)")

# count = zot.count_items()
# st.write('There are '+  '**'+str(count)+ '**' + ' items in the Zotero group library. To see the full library, go to the [Intelligence bibliography Zotero group library](https://www.zotero.org/groups/2514686/intelligence_bibliography/items).')
# df['Date added'] = pd.to_datetime(df['Date added'], errors='coerce')
# df['Date added'] = df['Date added'].dt.strftime('%d/%m/%Y')
# st.write('The library last updated on ' + '**'+ df.loc[0]['Date added']+'**')

# image = 'https://images.pexels.com/photos/315918/pexels-photo-315918.png'

# with st.sidebar:
#     st.image(image, width=150)
#     st.sidebar.markdown("# Intelligence bibliography")
#     with st.expander('About'):
#         st.write('''This website lists secondary sources on intelligence studies and intelligence history.
#         The sources are originally listed in the [Intelligence bibliography Zotero library](https://www.zotero.org/groups/2514686/intelligence_bibliography).
#         This website uses [Zotero API](https://github.com/urschrei/pyzotero) to connect the *Intelligence bibliography Zotero group library*.
#         To see more details about the sources, please visit the group library [here](https://www.zotero.org/groups/2514686/intelligence_bibliography/library). 
#         If you need more information about Zotero, visit [this page](https://www.intelligencenetwork.org/zotero).
#         ''')
#         components.html(
#         """
#         <a rel="license" href="http://creativecommons.org/licenses/by/4.0/"><img alt="Creative Commons Licence" style="border-width:0" 
#         src="https://i.creativecommons.org/l/by/4.0/80x15.png" /></a><br />
#         © 2022 All rights reserved. This website is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">Creative Commons Attribution 4.0 International License</a>.
#         """
#         )
#     with st.expander('Source code'):
#         st.info('''
#         Source code of this app is available [here](https://github.com/YusufAliOzkan/zotero-intelligence-bibliography).
#         ''')
#     with st.expander('Disclaimer'):
#         st.warning('''
#         This website and the Intelligence bibliography Zotero group library do not list all the sources on intelligence studies. 
#         The list is created based on the creator's subjective views.
#         ''')
#     with st.expander('Contact us'):
#         st.write('If you have any questions or suggestions, please do get in touch with us by filling the form [here](https://www.intelligencenetwork.org/contact-us).')

# # Recently added items

# col1, col2 = st.columns([4,2]) 
# with col1:
#     st.header('Recently added items: ')
#     with st.expander('Click to hide the list', expanded=True):
#         display = st.checkbox('Display abstract')

#         df_last = '**'+ df['Publication type']+ '**'+ ': ' +  df['Title'] + ' '+ "[[Publication link]]" +'('+ df['Link to publication'] + ')' +'  '+ "[[Zotero link]]" +'('+ df['Zotero link'] + ')' + ' (Added on: ' + df['Date added']+')'
#         row_nu_1= len(df_last.index)

#         for i in range(row_nu_1):
#             st.write(''+str(i+1)+') ' +df_last.iloc[i])
#             if display:
#                 st.caption('Abstract:'+'\n '+ df['Abstract'].iloc[i])

#     # Collection list
#     bbb = zot.collections()
#     data3=[]
#     columns3 = ['Key','Name', 'Number', 'Link']
#     for item in bbb:
#         data3.append((item['data']['key'], item['data']['name'], item['meta']['numItems'], item['links']['alternate']['href']))
#     pd.set_option('display.max_colwidth', None)
#     df_collections_2 = pd.DataFrame(data3, columns=columns3)

#     st.header('Items by collection: ')

#     collections = zot.collections()
#     data2=[]
#     columns2 = ['Key','Name', 'Number', 'Link']
#     for item in collections:
#         data2.append((item['data']['key'], item['data']['name'], item['meta']['numItems'], item['links']['alternate']['href']))

#     pd.set_option('display.max_colwidth', None)
#     df_collections = pd.DataFrame(data2, columns=columns2)
#     df_collections

#     df_collections = df_collections.sort_values(by='Name')

#     clist = df_collections['Name'].unique()
#     collection_name = st.selectbox('Select a collection:', clist)
#     collection_code = df_collections.loc[df_collections['Name']==collection_name, 'Key'].values[0]

#     df_collections=df_collections['Name'].reset_index()
#     pd.set_option('display.max_colwidth', None)

#     # Collection items

#     items = zot.everything(zot.collection_items_top(collection_code))

#     data3=[]
#     columns3=['Title','Publication type', 'Link to publication', 'Abstract', 'Zotero link']

#     for item in items:
#         data3.append((item['data']['title'], item['data']['itemType'], item['data']['url'], item['data']['abstractNote'], item['links']['alternate']['href'])) 
#     pd.set_option('display.max_colwidth', None)

#     df = pd.DataFrame(data3, columns=columns3)

#     df['Publication type'] = df['Publication type'].replace(['thesis'], 'Thesis')
#     df['Publication type'] = df['Publication type'].replace(['journalArticle'], 'Journal article')
#     df['Publication type'] = df['Publication type'].replace(['book'], 'Book')
#     df['Publication type'] = df['Publication type'].replace(['bookSection'], 'Book chapter')
#     df['Publication type'] = df['Publication type'].replace(['blogPost'], 'Blog post')
#     df['Publication type'] = df['Publication type'].replace(['videoRecording'], 'Video')
#     df['Publication type'] = df['Publication type'].replace(['podcast'], 'Podcast')
#     df['Publication type'] = df['Publication type'].replace(['magazineArticle'], 'Magazine article')
#     df['Publication type'] = df['Publication type'].replace(['webpage'], 'Webpage')
#     df['Publication type'] = df['Publication type'].replace(['newspaperArticle'], 'Newspaper article')
#     df['Publication type'] = df['Publication type'].replace(['report'], 'Report')

#     df_items = '**'+ df['Publication type']+ '**'+ ': ' +  df['Title'] + ' '+ "[[Publication link]]" +'('+ df['Link to publication'] + ')' +'  '+ "[[Zotero link]]" +'('+ df['Zotero link'] + ')'

#     row_nu_1= len(df_items.index)
#     if row_nu_1<15:
#         row_nu_1=row_nu_1
#     else:
#         row_nu_1=15

#     st.markdown('#### Collection theme: ' + collection_name)

#     with st.expander("Expand to see the list", expanded=False):
#         st.write('This list shows the last 15 added items. To see the full collection list click [here](https://www.zotero.org/groups/2514686/intelligence_bibliography/collections/' + collection_code + ')')
#         # display2 = st.checkbox('Display abstracts')
#         for i in range(row_nu_1):
#             st.write(''+str(i+1)+') ' +df_items.iloc[i])
#             df_items.fillna("nan") 
#             # if display2:
#             #     st.caption(df['Abstract'].iloc[i])

# with col2:
#     with st.expander("Collections in Zotero library", expanded=False):
#         row_nu_collections = len(df_collections_2.index)

#         for i in range(row_nu_collections):
#             st.caption('[' + df_collections_2.sort_values(by='Name')['Name'].iloc[i]+ ']'+ '('+ df_collections_2.sort_values(by='Name')['Link'].iloc[i] + ')')

#     # Zotero library collections

# components.html(
# """
# <a rel="license" href="http://creativecommons.org/licenses/by/4.0/"><img alt="Creative Commons Licence" style="border-width:0" 
# src="https://i.creativecommons.org/l/by/4.0/80x15.png" /></a><br />
# © 2022 All rights reserved. This website is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">Creative Commons Attribution 4.0 International License</a>.
# """
# )