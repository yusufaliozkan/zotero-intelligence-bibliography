# streamlit run "D:\OneDrive\06.Data_science\Zotero\zotero_app.py"

from pyzotero import zotero
import pandas as pd
import streamlit as st
# from bokeh.models.widgets import Button
# from bokeh.models import CustomJS
# from streamlit_bokeh_events import streamlit_bokeh_events

library_id = '2514686'
library_type = 'group'
api_key = '' # api_key is only needed for private groups and libraries

zot = zotero.Zotero(library_id, library_type)
items = zot.top(limit=10)
# items = zot.items()

# items = zot.collection_items_top('BNPYHVD4', limit=10)
pd.set_option('display.max_colwidth', None)

data=[]
columns = ['Title','Publication type', 'Link to publication', 'Abstract', 'Zotero link', 'Date added']

for item in items:
    data.append((item['data']['title'], item['data']['itemType'], item['data']['url'], item['data']['abstractNote'], item['links']['alternate']['href'], item['data']['dateAdded'],))

st.set_page_config(layout = "centered") 

df = pd.DataFrame(data, columns=columns)

# Change type name
df['Publication type'] = df['Publication type'].replace(['thesis'], 'Thesis')
df['Publication type'] = df['Publication type'].replace(['journalArticle'], 'Journal article')
df['Publication type'] = df['Publication type'].replace(['book'], 'Book')
df['Publication type'] = df['Publication type'].replace(['bookSection'], 'Book chapter')
df['Publication type'] = df['Publication type'].replace(['blogPost'], 'Blog post')
df['Publication type'] = df['Publication type'].replace(['videoRecording'], 'Video')
df['Publication type'] = df['Publication type'].replace(['podcast'], 'Podcast')
df['Publication type'] = df['Publication type'].replace(['magazineArticle'], 'Magazine article')


st.title("Intelligence bibliography - [Zotero group library](https://www.zotero.org/groups/2514686/intelligence_bibliography/library)")

count = zot.count_items()
st.write('There are '+  str(count) + ' items in the Zotero group library. To see the full library click [link](https://www.zotero.org/groups/2514686/intelligence_bibliography/items)')
df['Date added'] = pd.to_datetime(df['Date added'], errors='coerce')
df['Date added'] = df['Date added'].dt.strftime('%d/%m/%Y')
st.write('The library last updated on ' + df.loc[0]['Date added'])


# Recently added items

st.header('Recently added items: ')
display = st.checkbox('Display abstract')

df_last = '**'+ df['Publication type']+ '**'+ ': ' +  df['Title'] + ' '+ "[[Publication link]]" +'('+ df['Link to publication'] + ')' +'  '+ "[[Zotero link]]" +'('+ df['Zotero link'] + ')' + ' (Added on: ' + df['Date added']+')'
row_nu_1= len(df_last.index)

for i in range(row_nu_1):
    st.write(''+str(i+1)+') ' +df_last.iloc[i])
    if display:
        st.caption('Abstract:'+'\n '+ df['Abstract'].iloc[i])

# Zotero library collections
st.header('Items by collection: ')

col1, col2 =st.columns([3,5])

aaa = zot.collections()
data2=[]
columns2 = ['Code','Name', 'Number', 'Link', 'Parent collection']
for item in aaa:
    data2.append((item['data']['key'], item['data']['name'], item['meta']['numItems'], item['links']['alternate']['href'], item['data']['parentCollection']))

pd.set_option('display.max_colwidth', None)
df_collections = pd.DataFrame(data2, columns=columns2)

# df_collections = df_collections.sort_values(by='Name')


# df_collections['Name2'] = df_collections['Name'].replace(['01. Intelligence history'],'Intelligence history')
# df_collections['Name2'] = df_collections['Name2'].replace(['02. Intelligence studies'],'Intelligence studies')
# df_collections['Name2'] = df_collections['Name2'].replace(['03. Intelligence analysis'],'Intelligence analysis')
# df_collections['Name2'] = df_collections['Name2'].replace(['04. Intelligence organisations'],'Intelligence organisations')
# df_collections['Name2'] = df_collections['Name2'].replace(['05. Intelligence failures'],'Intelligence failures')
# df_collections['Name2'] = df_collections['Name2'].replace(['06. Intelligence, oversight, and ethics'],'Intelligence, oversight, and ethics')
# df_collections['Name2'] = df_collections['Name2'].replace(['07. a. HUMINT'],'HUMINT')
# df_collections['Name2'] = df_collections['Name2'].replace(['07. b. IMINT'],'IMINT')
# df_collections['Name2'] = df_collections['Name2'].replace(['07. c. OSINT - SOCMINT'],'OSINT - SOCMINT')
# df_collections['Name2'] = df_collections['Name2'].replace(['07. d. SIGINT'],'SIGINT')
# df_collections['Name2'] = df_collections['Name2'].replace(['07. e. Medical Intelligence'],'Medical Intelligence')
# df_collections['Name2'] = df_collections['Name2'].replace(['08. Counterintelligence'],'Counterintelligence')
# df_collections['Name2'] = df_collections['Name2'].replace(['09. Covert action'],'Covert action')
# df_collections['Name2'] = df_collections['Name2'].replace(['10. Cybersphere'],'Cybersphere')
# df_collections['Name2'] = df_collections['Name2'].replace(['11. Intelligence in literature and popular culture'],'Intelligence in literature and popular culture')
# df_collections['Name2'] = df_collections['Name2'].replace(['12. Documentaries and movies'],'Documentaries and movies')
# df_collections['Name2'] = df_collections['Name2'].replace(['13. Podcasts and videos'],'Podcasts and videos')
# df_collections['Name2'] = df_collections['Name2'].replace(['14. Miscellaneous'],'Miscellaneous')
# df_collections['Name2'] = df_collections['Name2'].replace(['15. Methodology'],'Methodology')
# df_collections['Name2'] = df_collections['Name2'].replace(['98. Special collections'],'Special collections')
# df_collections['Name2'] = df_collections['Name2'].replace(['99. Archival sources and reports'],'Archival sources and reports')

clist = df_collections['Name'].unique()
collection_name = st.selectbox('Select a collection:', clist)
collection_code = df_collections.loc[df_collections['Name']==collection_name, 'Code'].values[0]

df_collections=df_collections['Name'].reset_index()
pd.set_option('display.max_colwidth', None)

st.dataframe(df_collections['Code'], height=1500)

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
df['Publication type'] = df['Publication type'].replace(['newspaperArticle'], 'Newspaper article')
df['Publication type'] = df['Publication type'].replace(['report'], 'Report')
df['Publication type'] = df['Publication type'].replace(['webpage'], 'Webpage')

df_items = '**'+ df['Publication type']+ '**'+ ': ' +  df['Title'] + ' '+ "[[Publication link]]" +'('+ df['Link to publication'] + ')' +'  '+ "[[Zotero link]]" +'('+ df['Zotero link'] + ')'

row_nu_1= len(df_items.index)
if row_nu_1<25:
    row_nu_1=row_nu_1
else:
    row_nu_1=25

st.markdown('#### Collection theme: ' + collection_name)

with st.expander("Expand to see the list", expanded=False):
    st.write('This list shows the last 25 added items. To see the full collection list click [here](https://www.zotero.org/groups/2514686/intelligence_bibliography/collections/' + collection_code + ')')
    # display2 = st.checkbox('Display abstracts')
    for i in range(row_nu_1):
        st.write(''+str(i+1)+') ' +df_items.iloc[i])
        df_items.fillna("nan") 
        # if display2:
        #     st.caption(df['Abstract'].iloc[i])


