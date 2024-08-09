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
import datetime as dt
import base64
from sidebar_content import sidebar_content
from format_entry import format_entry
from streamlit_gsheets import GSheetsConnection


st.set_page_config(layout = "centered", 
                    page_title='Intelligence studies network',
                    page_icon="https://images.pexels.com/photos/315918/pexels-photo-315918.png",
                    initial_sidebar_state="auto") 

st.title("Intelligence studies network")
st.header('Digest')

with st.spinner('Preparing digest...'):

    sidebar_content()

    df_csv = pd.read_csv(r'all_items.csv', index_col=None)
    df_csv['Date published'] = (
        df_csv['Date published']
        .str.strip()
        .apply(lambda x: pd.to_datetime(x, utc=True, errors='coerce').tz_convert('Europe/London'))
    )
    df_csv['Date published'] = pd.to_datetime(df_csv['Date published'],utc=True, errors='coerce').dt.date
    df_csv['Publisher'] =df_csv['Publisher'].fillna('')
    df_csv['Journal'] =df_csv['Journal'].fillna('')
    df_csv['FirstName2'] =df_csv['FirstName2'].fillna('')

    df_csv = df_csv.drop(['Unnamed: 0'], axis=1)
    df_cited = df_csv.copy()

    today = dt.date.today()
    today2 = dt.date.today().strftime('%d/%m/%Y')
    st.write('Your daily intelligence studies digest - Day: '+ str(today2))

    st.markdown('#### Contents')
    st.caption('[Publications](#publications)')
    st.caption('[Events](#events)')
    st.caption('[Conferences](#conferences)')
    st.caption('[Call for papers](#call-for-papers)')

    ex=False
    expand = st.checkbox('Expand all', key='expand')
    if expand:
        ex = True

    st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)

    with st.expander('Publications:', expanded=ex):
        st.header('Publications')
        options = st.radio('Select an option', ('Recently added', 'Recently published', 'Recently cited'))

        if options=='Recently added':
            previous_7 = today - dt.timedelta(days=7)
            previous_30 = today - dt.timedelta(days=30)
            # previous_180 = today - dt.timedelta(days=180)
            # previous_360 = today - dt.timedelta(days=365)
            rg = previous_7
            a='30 days'

            df_csv['Date added'] = pd.to_datetime(df_csv['Date added'], errors='coerce').dt.date
            latest_added_date = df_csv['Date added'].max()

            current_year = today.year
            if today.month == 12:
                start_last_month = dt.date(current_year, 11, 1)
                end_last_month = dt.date(current_year, 11, 30)
            else:
                start_last_month = dt.date(current_year, 12, 1)
                end_last_month = dt.date(current_year, 12, 31)

            range_day = st.radio('Show sources added to the database in the last:', ('7 days','30 days', 'Custom (select date)'), key='days_recently_added')
            if range_day == '7 days':
                rg = previous_7
                a='7 days'
            if range_day == '30 days':
                rg = previous_30
                a = '30 days'
            if range_day == '3 months':
                rg = previous_180
                a ='3 months'
            if range_day == 'Custom (days)':
                number = st.number_input('How many days do you want to go back:', min_value=10, max_value=11000, value=365, step=30)
                a = str(int(number)) + ' days'
                previous_custom = today - dt.timedelta(days=number)
                rg = previous_custom
            if range_day == 'Custom (select date)':
                rg = st.date_input('From:', today-dt.timedelta(days=7), max_value=(latest_added_date))
                today = st.date_input('To:', today, max_value=today, min_value=rg)
                a = today - rg
                a = str(a.days) + ' days'

            if range_day == 'Custom (select date)':
                filter = (df_csv['Date added']>=rg) & (df_csv['Date added']<=today)
            else:
                filter = (df_csv['Date added']>rg) & (df_csv['Date added']<=today)
            rg2 = rg.strftime('%d/%m/%Y')
            df_csv = df_csv.loc[filter]

            df_csv['Date published'] = pd.to_datetime(df_csv['Date published'],utc=True, errors='coerce').dt.tz_convert('Europe/London')
            df_csv['Date published new'] = df_csv['Date published'].dt.strftime('%d/%m/%Y')
            df_csv['Date months'] = df_csv['Date published'].dt.strftime('%Y-%m')
            df_csv['Date published'] = df_csv['Date published'].fillna('No date')
            # df_csv.sort_values(by='Date published', ascending = False, inplace=True)    

            sort_by_type = st.checkbox('Sort by publication type', key='type')
            st.caption('See [📊 trends](#trends) in the last ' + str(a))
            types = st.multiselect('Publication type', df_csv['Publication type'].unique(),df_csv['Publication type'].unique())
            df_csv = df_csv[df_csv['Publication type'].isin(types)]
            df_csv["Link to publication"].fillna("No link", inplace = True)
            if range_day == 'Custom (select date)':
                num_items = len(df_csv)
                difference = (today-rg)
                days_difference = difference.days
                today_2 = today.strftime('%d/%m/%Y')
                st.subheader('Sources added between ' + '**'+ rg2 +' - ' + today_2+'**')
                st.write(f"**{num_items}** sources added in the last {days_difference} day" if days_difference == 1 else f"**{num_items}** sources added in the last {days_difference} days")
            else:
                num_items = len(df_csv)
                st.subheader('Sources added in the last ' + str(a))
                difference = (today-rg)
                days_difference = difference.days
                st.write(f"**{num_items}** sources added in the last {days_difference} day" if days_difference == 1 else f"**{num_items}** sources added in the last {days_difference} days")

            if df_csv['Title'].any() in ("", [], None, 0, False):
                st.write('There is no publication added in the last '+ str(a))

            df_csv['Date published'] = pd.to_datetime(df_csv['Date published'], errors='coerce').dt.date
            if sort_by_type:
                df_csv = df_csv.sort_values(by=['Publication type'], ascending=True)
                current_type = None
                count_by_type = {}
                for index, row in df_csv.iterrows():
                    if row['Publication type'] != current_type:
                        current_type = row['Publication type']
                        st.subheader(current_type)
                        count_by_type[current_type] = 1
                    formatted_entry = format_entry(row)
                    st.write(f"{count_by_type[current_type]}) {formatted_entry}")
                    count_by_type[current_type] += 1

            else:
                row_nu99 = len(df_csv)
                if row_nu99 == 0:
                    st.write('Select a publication type')
                else:
                    articles_list = []  # Store articles in a list
                    for index, row in df_csv.iterrows():
                        formatted_entry = format_entry(row)  # Assuming format_entry() is a function formatting each row
                        articles_list.append(formatted_entry)        
                    
                    for index, row in df_csv.iterrows():
                        publication_type = row['Publication type']
                        title = row['Title']
                        authors = row['FirstName2']
                        date_published = row['Date published new']
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
                        '(Publication date: ' + (date_published) + ') ' +
                        ('(' + published_by_or_in + ': ' + '*' + str(published_source) + '*' + ') ' if published_by_or_in else '') +
                        '[[Publication link]](' + str(link_to_publication) + ') ' +
                        '[[Zotero link]](' + str(zotero_link) + ')'
                    )
                    count = 1
                    for index, row in df_csv.iterrows():
                        formatted_entry = format_entry(row)
                        st.write(f"{count}) {formatted_entry}")
                        count += 1
            st.subheader('📊 Trends')
            if df_csv['Publication type'].any() in ("", [], None, 0, False):
                st.write('No data to visualise')
            else:
                trends = st.toggle('Show trends', key='trends')
                if trends:
                    df_plot= df_csv['Publication type'].value_counts()
                    df_plot=df_plot.reset_index()
                    df_plot=df_plot.rename(columns={'index':'Publication type','Publication type':'Count'})
                    df_plot.columns = ['Publication type', 'Count']
                    fig = px.bar(df_plot, x='Publication type', y='Count', color='Publication type')
                    fig.update_layout(
                        autosize=False,
                        width=400,
                        height=400,)
                    fig.update_layout(title={'text':'Publication types in the last '+a+' ('+ rg2 +' - ' + today2+')', 'y':0.95, 'x':0.2, 'yanchor':'top'})
                    st.plotly_chart(fig, use_container_width = True)

                    df_csv['Date published'] = df_csv['Date published'].dt.strftime('%Y-%m-%d')
                    df_dates = df_csv['Date published'].value_counts()
                    df_dates = df_dates.reset_index()
                    df_dates = df_dates.rename(columns={'index':'Publication date','Date published':'Count'})
                    df_dates.columns = ['Publication date', 'Count']
                    df_dates = df_dates.sort_values(by='Publication date', ascending=True)
                    df_dates['sum'] = df_dates['Count'].cumsum()

                    df_months = df_csv['Date months'].value_counts()
                    df_months = df_months.reset_index()
                    df_months = df_months.rename(columns={'index':'Publication month','Date months':'Count'})
                    df_months.columns = ['Publication month', 'Count']
                    df_months = df_months.sort_values(by='Publication month', ascending=True)
                    df_months['sum'] = df_months['Count'].cumsum()

                    if range_day == '6 months' or range_day == '1 year' or range_day == 'Custom':
                        fig = px.bar(df_months, x='Publication month', y='Count')
                        fig.update_xaxes(tickangle=-70)
                        fig.update_layout(
                            autosize=False,
                            width=400,
                            height=500,)
                        fig.update_layout(title={'text':'Publications by date in the last '+a+' ('+ rg2 +' - ' + today2+')', 'y':0.95, 'x':0.5, 'yanchor':'top'})
                        st.plotly_chart(fig, use_container_width = True)            
                    else:
                        fig = px.bar(df_dates, x='Publication date', y='Count')
                        fig.update_xaxes(tickangle=-70)
                        fig.update_layout(
                            autosize=False,
                            width=400,
                            height=500,)
                        fig.update_layout(title={'text':'Publications by date in the last '+a+' ('+ rg2 +' - ' + today2+')', 'y':0.95, 'x':0.5, 'yanchor':'top'})
                        st.plotly_chart(fig, use_container_width = True)

                    fig2 = px.line(df_dates, x='Publication date', y='sum')
                    fig2.update_layout(title={'text':'Publications by date in the last '+a+' ('+ rg2 +' - ' + today2+')'+ ' (cumulative sum)', 'y':0.95, 'x':0.5, 'yanchor':'top'})
                    fig2.update_xaxes(tickangle=-70)
                    st.plotly_chart(fig2, use_container_width = True)

                    df=df_csv.copy()
                    def clean_text (text):
                        text = text.lower() # lowercasing
                        text = re.sub(r'[^\w\s]', ' ', text) # this removes punctuation
                        text = re.sub('[0-9_]', ' ', text) # this removes numbers
                        text = re.sub('[^a-z_]', ' ', text) # removing all characters except lowercase letters
                        return text
                    df['clean_title'] = df['Title'].apply(clean_text)
                    df['clean_title'] = df['clean_title'].apply(lambda x: ' '.join ([w for w in x.split() if len (w)>2])) # this function removes words less than 2 words

                    def tokenization(text):
                        text = re.split('\W+', text)
                        return text
                    df['token_title']=df['clean_title'].apply(tokenization)
                    stopword = nltk.corpus.stopwords.words('english')

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

                    wn = nltk.WordNetLemmatizer()
                    def lemmatizer(text):
                        text = [wn.lemmatize(word) for word in text]
                        return text

                    df['lemma_title'] = df['stopword'].apply(lemmatizer) # error occurs in this line

                    listdf = df['lemma_title']

                    df_list = [item for sublist in listdf for item in sublist]
                    string = pd.Series(df_list).str.cat(sep=' ')
                    wordcloud_texts = string
                    wordcloud_texts_str = str(wordcloud_texts)
                    wordcloud = WordCloud(stopwords=stopword, width=1500, height=750, background_color='white', collocations=False, colormap='magma').generate(wordcloud_texts_str)
                    plt.figure(figsize=(20,8))
                    plt.axis('off')
                    plt.title('Top words of titles published in the last ' +a+' ('+ rg2 +' - ' + today2+')')
                    plt.imshow(wordcloud)
                    plt.axis("off")
                    plt.show()
                    st.set_option('deprecation.showPyplotGlobalUse', False)
                    st.pyplot() 
        elif options=='Recently published':
            previous_10 = today - dt.timedelta(days=10)
            previous_30 = today - dt.timedelta(days=30)
            previous_180 = today - dt.timedelta(days=180)
            previous_360 = today - dt.timedelta(days=365)
            rg = previous_10
            a='30 days'
            st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)


            range_day = st.radio('Show sources published in the last:', ('10 days','30 days', '3 months', 'Custom (days)', 'Custom (select date)'), key='days')
            if range_day == '10 days':
                rg = previous_10
                a='10 days'
            if range_day == '30 days':
                rg = previous_30
                a = '30 days'
            if range_day == '3 months':
                rg = previous_180
                a ='3 months'
            if range_day == 'Custom (days)':
                number = st.number_input('How many days do you want to go back:', min_value=10, max_value=11000, value=365, step=30)
                a = str(int(number)) + ' days'
                previous_custom = today - dt.timedelta(days=number)
                rg = previous_custom
            if range_day == 'Custom (select date)':
                rg = st.date_input('From:', today-dt.timedelta(days=7), max_value=today-dt.timedelta(days=0))
                today = st.date_input('To:', today, max_value=today, min_value=rg)
                a = today - rg
                a = str(a.days) + ' days'

            filter = (df_csv['Date published']>rg) & (df_csv['Date published']<=today)
            rg2 = rg.strftime('%d/%m/%Y')
            df_csv = df_csv.loc[filter]

            df_csv['Date published'] = pd.to_datetime(df_csv['Date published'],utc=True, errors='coerce').dt.tz_convert('Europe/London')

            df_csv['Date published new'] = df_csv['Date published'].dt.strftime('%d/%m/%Y')
            df_csv['Date months'] = df_csv['Date published'].dt.strftime('%Y-%m')
            df_csv['Date published'] = df_csv['Date published'].fillna('No date')
            df_csv.sort_values(by='Date published', ascending = False, inplace=True)    

            sort_by_type = st.checkbox('Sort by publication type', key='type')
            st.caption('See [📊 trends](#trends) in the last ' + str(a))
            types = st.multiselect('Publication type', df_csv['Publication type'].unique(),df_csv['Publication type'].unique())
            df_csv = df_csv[df_csv['Publication type'].isin(types)]
            df_csv["Link to publication"].fillna("No link", inplace = True)
            if range_day == 'Custom (select date)':
                num_items = len(df_csv)
                today_2 = today.strftime('%d/%m/%Y')
                st.subheader('Sources published between ' + '**'+ rg2 +' - ' + today_2+'**')
                st.write('This list finds '+str(num_items)+' sources published between ' + '**'+ rg2 +' - ' + today_2+'**')
            else:
                num_items = len(df_csv)
                st.subheader('Sources published in the last ' + str(a))
                st.write('This list finds '+str(num_items)+' sources published between ' + '**'+ rg2 +' - ' + today2+'**')    

            if df_csv['Title'].any() in ("", [], None, 0, False):
                st.write('There is no publication published in the last '+ str(a))

            df_csv['Date published'] = pd.to_datetime(df_csv['Date published'], errors='coerce').dt.date
            if sort_by_type:
                df_csv = df_csv.sort_values(by=['Publication type'], ascending=True)
                current_type = None
                count_by_type = {}
                for index, row in df_csv.iterrows():
                    if row['Publication type'] != current_type:
                        current_type = row['Publication type']
                        st.subheader(current_type)
                        count_by_type[current_type] = 1
                    formatted_entry = format_entry(row)
                    st.write(f"{count_by_type[current_type]}) {formatted_entry}")
                    count_by_type[current_type] += 1

            else:
                row_nu99 = len(df_csv)
                if row_nu99 == 0:
                    st.write('Select a publication type')
                else:
                    articles_list = []  # Store articles in a list
                    for index, row in df_csv.iterrows():
                        formatted_entry = format_entry(row)  # Assuming format_entry() is a function formatting each row
                        articles_list.append(formatted_entry)        
                    
                    for index, row in df_csv.iterrows():
                        publication_type = row['Publication type']
                        title = row['Title']
                        authors = row['FirstName2']
                        date_published = row['Date published new']
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
                        '(Publication date: ' + (date_published) + ') ' +
                        ('(' + published_by_or_in + ': ' + '*' + str(published_source) + '*' + ') ' if published_by_or_in else '') +
                        '[[Publication link]](' + str(link_to_publication) + ') ' +
                        '[[Zotero link]](' + str(zotero_link) + ')'
                    )
                    count = 1
                    for index, row in df_csv.iterrows():
                        formatted_entry = format_entry(row)
                        st.write(f"{count}) {formatted_entry}")
                        count += 1

            st.subheader('📊 Trends')
            if df_csv['Publication type'].any() in ("", [], None, 0, False):
                st.write('No data to visualise')
            else:
                trends = st.toggle('Show trends', key='trends')
                if trends:
                    df_plot= df_csv['Publication type'].value_counts()
                    df_plot=df_plot.reset_index()
                    df_plot=df_plot.rename(columns={'index':'Publication type','Publication type':'Count'})
                    df_plot.columns = ['Publication type', 'Count']
                    fig = px.bar(df_plot, x='Publication type', y='Count', color='Publication type')
                    fig.update_layout(
                        autosize=False,
                        width=400,
                        height=400,)
                    fig.update_layout(title={'text':'Publication types in the last '+a+' ('+ rg2 +' - ' + today2+')', 'y':0.95, 'x':0.2, 'yanchor':'top'})
                    st.plotly_chart(fig, use_container_width = True)

                    df_csv['Date published'] = df_csv['Date published'].dt.strftime('%Y-%m-%d')
                    df_dates = df_csv['Date published'].value_counts()
                    df_dates = df_dates.reset_index()
                    df_dates = df_dates.rename(columns={'index':'Publication date','Date published':'Count'})
                    df_dates.columns = ['Publication date', 'Count']
                    df_dates = df_dates.sort_values(by='Publication date', ascending=True)
                    df_dates['sum'] = df_dates['Count'].cumsum()

                    df_months = df_csv['Date months'].value_counts()
                    df_months = df_months.reset_index()
                    df_months = df_months.rename(columns={'index':'Publication month','Date months':'Count'})
                    df_months.columns = ['Publication month', 'Count']
                    df_months = df_months.sort_values(by='Publication month', ascending=True)
                    df_months['sum'] = df_months['Count'].cumsum()

                    if range_day == '6 months' or range_day == '1 year' or range_day == 'Custom':
                        fig = px.bar(df_months, x='Publication month', y='Count')
                        fig.update_xaxes(tickangle=-70)
                        fig.update_layout(
                            autosize=False,
                            width=400,
                            height=500,)
                        fig.update_layout(title={'text':'Publications by date in the last '+a+' ('+ rg2 +' - ' + today2+')', 'y':0.95, 'x':0.5, 'yanchor':'top'})
                        st.plotly_chart(fig, use_container_width = True)            
                    else:
                        fig = px.bar(df_dates, x='Publication date', y='Count')
                        fig.update_xaxes(tickangle=-70)
                        fig.update_layout(
                            autosize=False,
                            width=400,
                            height=500,)
                        fig.update_layout(title={'text':'Publications by date in the last '+a+' ('+ rg2 +' - ' + today2+')', 'y':0.95, 'x':0.5, 'yanchor':'top'})
                        st.plotly_chart(fig, use_container_width = True)

                    fig2 = px.line(df_dates, x='Publication date', y='sum')
                    fig2.update_layout(title={'text':'Publications by date in the last '+a+' ('+ rg2 +' - ' + today2+')'+ ' (cumulative sum)', 'y':0.95, 'x':0.5, 'yanchor':'top'})
                    fig2.update_xaxes(tickangle=-70)
                    st.plotly_chart(fig2, use_container_width = True)

                    df=df_csv.copy()
                    def clean_text (text):
                        text = text.lower() # lowercasing
                        text = re.sub(r'[^\w\s]', ' ', text) # this removes punctuation
                        text = re.sub('[0-9_]', ' ', text) # this removes numbers
                        text = re.sub('[^a-z_]', ' ', text) # removing all characters except lowercase letters
                        return text
                    df['clean_title'] = df['Title'].apply(clean_text)
                    df['clean_title'] = df['clean_title'].apply(lambda x: ' '.join ([w for w in x.split() if len (w)>2])) # this function removes words less than 2 words

                    def tokenization(text):
                        text = re.split('\W+', text)
                        return text
                    df['token_title']=df['clean_title'].apply(tokenization)
                    stopword = nltk.corpus.stopwords.words('english')

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

                    wn = nltk.WordNetLemmatizer()
                    def lemmatizer(text):
                        text = [wn.lemmatize(word) for word in text]
                        return text

                    df['lemma_title'] = df['stopword'].apply(lemmatizer) # error occurs in this line

                    listdf = df['lemma_title']

                    df_list = [item for sublist in listdf for item in sublist]
                    string = pd.Series(df_list).str.cat(sep=' ')
                    wordcloud_texts = string
                    wordcloud_texts_str = str(wordcloud_texts)
                    wordcloud = WordCloud(stopwords=stopword, width=1500, height=750, background_color='white', collocations=False, colormap='magma').generate(wordcloud_texts_str)
                    plt.figure(figsize=(20,8))
                    plt.axis('off')
                    plt.title('Top words of titles published in the last ' +a+' ('+ rg2 +' - ' + today2+')')
                    plt.imshow(wordcloud)
                    plt.axis("off")
                    plt.show()
                    st.set_option('deprecation.showPyplotGlobalUse', False)
                    st.pyplot() 
        elif options=='Recently cited':
            current_year = datetime.datetime.now().year
            df_cited = df_cited[(df_cited['Citation'].notna()) & (df_cited['Citation'] != 0)]
            df_cited = df_cited.reset_index(drop=True)
            df_cited = df_cited[(df_cited['Last_citation_year'] == current_year) | (df_cited['Last_citation_year'] == current_year)]
            df_cited = df_cited.reset_index(drop=True)
            if len(df_cited) == 0: 
                st.warning(f'No citation yet in {current_year}') 
            else:
                row_nu99 = len(df_cited)
                st.info(f'**{row_nu99} paper(s) cited in {current_year}**.')
                articles_list = []  # Store articles in a list
                for index, row in df_cited.iterrows():
                    formatted_entry = format_entry(row)  # Assuming format_entry() is a function formatting each row
                    articles_list.append(formatted_entry)        
                
                for index, row in df_cited.iterrows():
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
                sort_by = st.radio('Sort by:', ('Publication date :arrow_down:', 'Citation'), key=123)
                display2 = st.checkbox('Display abstracts', key=1234)
                if sort_by == 'Publication date :arrow_down:' or df_cited['Citation'].sum() == 0:
                    count = 1
                    df_cited = df_cited.sort_values(by=['Date published'], ascending=False)
                    for index, row in df_cited.iterrows():
                        formatted_entry = format_entry(row)
                        st.write(f"{count}) {formatted_entry}")
                        count += 1
                        if display2:
                            st.caption(row['Abstract']) 
                else:
                    df_cited = df_cited.sort_values(by=['Citation'], ascending=False)
                    count = 1
                    for index, row in df_cited.iterrows():
                        formatted_entry = format_entry(row)
                        st.write(f"{count}) {formatted_entry}")
                        count += 1
                        if display2:
                            st.caption(row['Abstract']) 
        st.caption('[Go to top](#intelligence-studies-network-digest)')

    with st.expander('Events:', expanded=False):
        today = dt.date.today()
        today_datetime = pd.to_datetime(today)

        st.header('Events')
        conn = st.connection("gsheets", type=GSheetsConnection)
        df_gs = conn.read(spreadsheet='https://docs.google.com/spreadsheets/d/10ezNUOUpzBayqIMJWuS_zsvwklxP49zlfBWsiJI6aqI/edit#gid=0')

        df_forms = conn.read(spreadsheet='https://docs.google.com/spreadsheets/d/10ezNUOUpzBayqIMJWuS_zsvwklxP49zlfBWsiJI6aqI/edit#gid=1941981997')
        df_forms = df_forms.rename(columns={'Event name': 'event_name', 'Event organiser': 'organiser', 'Link to the event': 'link', 'Date of event': 'date', 'Event venue': 'venue', 'Details': 'details'})

        # Convert and format dates in df_gs
        df_gs['date'] = pd.to_datetime(df_gs['date'], format='%d/%m/%Y', errors='coerce')
        df_gs['date_new'] = df_gs['date'].dt.strftime('%Y-%m-%d')

        # Convert and format dates in df_forms
        df_forms['date'] = pd.to_datetime(df_forms['date'], format='%d/%m/%Y', errors='coerce')
        df_forms['date_new'] = df_forms['date'].dt.strftime('%Y-%m-%d')
        df_forms['month'] = df_forms['date'].dt.strftime('%m')
        df_forms['year'] = df_forms['date'].dt.strftime('%Y')
        df_forms['month_year'] = df_forms['date'].dt.strftime('%Y-%m')
        df_forms.sort_values(by='date', ascending=True, inplace=True)
        df_forms = df_forms.drop_duplicates(subset=['event_name', 'link', 'date'], keep='first')

        next_10 = today_datetime + dt.timedelta(days=10)
        next_20 = today_datetime + dt.timedelta(days=20)
        next_30 = today_datetime + dt.timedelta(days=30)
        rg2 = next_10
        aa = '10 days'
        range_day = st.radio('Show events in the next:', ('10 days', '20 days', '30 days'), key='events')
        
        if range_day == '10 days':
            rg2 = next_10
            aa = '10 days'
        elif range_day == '20 days':
            rg2 = next_20
            aa = '20 days'
        elif range_day == '30 days':
            rg2 = next_30
            aa = '30 days'

        # Filter events between today and the selected range day
        filter_events = (df_gs['date'] < rg2) & (df_gs['date'] >= today_datetime)
        df_gs = df_gs.loc[filter_events]

        st.subheader('Events in the next ' + str(aa))
        display = st.checkbox('Show details')
        if df_gs['event_name'].any() in ("", [], None, 0, False):
            st.write('No upcoming event in the next '+ str(a))
        df_gs1 = ('['+ df_gs['event_name'] + ']'+ '('+ df_gs['link'] + ')'', organised by ' + '**' + df_gs['organiser'] + '**' + '. Date: ' + df_gs['date_new'] + ', Venue: ' + df_gs['venue'])
        row_nu = len(df_gs.index)
        for i in range(row_nu):
            st.write(''+str(i+1)+') '+ df_gs1.iloc[i])
            if display:
                st.caption('Details:'+'\n '+ df_gs['details'].iloc[i])
        st.write('Visit the [Events on intelligence](https://intelligence.streamlit.app/Events) page to see more!')


        st.caption('[Go to top](#intelligence-studies-network-digest)')

    with st.expander('Conferences:', expanded=ex):
        st.header('Conferences')
        df_con = conn.read(spreadsheet='https://docs.google.com/spreadsheets/d/10ezNUOUpzBayqIMJWuS_zsvwklxP49zlfBWsiJI6aqI/edit#gid=939232836')
        df_con['date'] = pd.to_datetime(df_con['date'])
        df_con['date_new'] = df_con['date'].dt.strftime('%Y-%m-%d')
        df_con['date_new'] = pd.to_datetime(df_con['date'], dayfirst = True).dt.strftime('%d/%m/%Y')
        df_con['date_new_end'] = pd.to_datetime(df_con['date_end'], dayfirst = True).dt.strftime('%d/%m/%Y')
        df_con.sort_values(by='date', ascending = True, inplace=True)
        df_con['details'] = df_con['details'].fillna('No details')
        df_con['location'] = df_con['location'].fillna('No details')
        df_con = df_con.fillna('')
        df_con['date_end'] = pd.to_datetime(df_con['date'], dayfirst=True)    

        next_1mo = today_datetime  + dt.timedelta(days=30)
        next_3mo = today_datetime  + dt.timedelta(days=90)    
        next_6mo = today_datetime  + dt.timedelta(days=180)
        rg3 = next_3mo

        range_day = st.radio('Show conferences in the next: ', ('1 month', '3 months', '6 months'), key='conferences')
        if range_day == '1 month':
            rg3 = next_1mo
            aaa = '1 month'
        if range_day == '3 months':
            rg3 = next_3mo
            aaa = '3 months'
        if range_day == '6 months':
            rg3 = next_6mo
            aaa = '6 months'
        filter_events = (df_con['date'] < rg3) & (df_con['date'] >= today_datetime)
        df_con = df_con.loc[filter_events]

        df_con['details'] = df_con['details'].fillna('No details')
        df_con['location'] = df_con['location'].fillna('No details')

        st.subheader('Conferences in the next ' + str(aaa))
        display = st.checkbox('Show details', key='conference')
        if df_con['conference_name'].any() in ("", [], None, 0, False):
            st.write('No upcoming conference in the next '+ str(aaa))
        df_con1 = ('['+ df_con['conference_name'] + ']'+ '('+ df_con['link'] + ')'', organised by ' + '**' + df_con['organiser'] + '**' + '. Date(s): ' + df_con['date_new'] + ' - ' + df_con['date_new_end'] + ', Venue: ' + df_con['venue'])
        row_nu = len(df_con.index)
        for i in range(row_nu):
            st.write(''+str(i+1)+') '+ df_con1.iloc[i])
            if display:
                st.caption('Conference place:'+'\n '+ df_con['location'].iloc[i])
                st.caption('Details:'+'\n '+ df_con['details'].iloc[i])

        st.caption('[Go to top](#intelligence-studies-network-digest)')

    with st.expander('Call for papers:', expanded=ex):
        st.header('Call for papers')
        df_cfp = conn.read(spreadsheet='https://docs.google.com/spreadsheets/d/10ezNUOUpzBayqIMJWuS_zsvwklxP49zlfBWsiJI6aqI/edit#gid=135096406') 

        df_cfp['deadline'] = pd.to_datetime(df_cfp['deadline'])
        df_cfp['deadline_new'] = df_cfp['deadline'].dt.strftime('%d/%m/%Y')
        df_cfp.sort_values(by='deadline', ascending = True, inplace=True)

        df_cfp['details'] = df_cfp['details'].fillna('No details')
        df_cfp = df_cfp.fillna('')

        df_cfp = df_cfp.drop_duplicates(subset=['name', 'link', 'deadline'], keep='first')
        
        display = st.checkbox('Show details', key='cfp')
        df_cfp['deadline'] = pd.to_datetime(df_cfp['deadline'], dayfirst=True)

        filter = df_cfp['deadline']>=pd.to_datetime(today)
        df_cfp = df_cfp.loc[filter]
        if df_cfp['name'].any() in ("", [], None, 0, False):
            st.write('No upcoming Call for papers!')

        df_cfp1 = ('['+ df_cfp['name'] + ']'+ '('+ df_cfp['link'] + ')'', organised by '  + df_cfp['organiser'] + '. ' +'**' + 'Deadline: ' + df_cfp['deadline_new']+'**' )
        row_nu = len(df_cfp.index)
        for i in range(row_nu):
            st.write(''+str(i+1)+') '+ df_cfp1.iloc[i])
            if display:
                st.caption('Details:'+'\n '+ df_cfp['details'].iloc[i])

    st.caption('[Go to top](#intelligence-studies-network-digest)')

    st.write('---')

    components.html(
    """
    <a rel="license" href="http://creativecommons.org/licenses/by/4.0/"><img alt="Creative Commons Licence" style="border-width:0" 
    src="https://i.creativecommons.org/l/by/4.0/80x15.png" /></a><br />
    © 2024 Yusuf Ozkan. All rights reserved. This website is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">Creative Commons Attribution 4.0 International License</a>.
    """
    ) 
