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
from fpdf import FPDF
import base64


st.set_page_config(layout = "centered", 
                    page_title='Intelligence studies network',
                    page_icon="https://images.pexels.com/photos/315918/pexels-photo-315918.png",
                    initial_sidebar_state="auto") 

st.title("Intelligence studies network")
st.header('Institutions & programs on intelligence')

with st.spinner('Preparing...'):

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

    conn = connect()

    # Perform SQL query on the Google Sheet.
    # Uses st.cache to only rerun when the query changes or after 10 min.
    @st.cache_resource(ttl=10)
    def run_query(query):
        rows = conn.execute(query, headers=1)
        rows = rows.fetchall()
        return rows

    sheet_url = st.secrets["public_gsheets_url_orgs"]
    rows = run_query(f'SELECT * FROM "{sheet_url}"')

    data = []
    columns = ['Type', 'Institution', 'Programme_level', 'Programme_name', 'Link', 'Country', 'Status']

    # Print results.
    for row in rows:
        data.append((row.Type, row.Institution, row.Programme_level, row.Programme_name, row.Link, row.Country, row.Status))
    df_gs = pd.DataFrame(data, columns=columns)
    countries = df_gs['Country'].unique()


    for country in countries:
        with st.expander(f"Institutions in {country}"):
            filtered_data = df_gs[df_gs['Country'] == country]
            counter = 1
            for index, row in filtered_data.iterrows():
                if row['Programme_level'] is not None and row['Programme_name'] is not None:
                    st.write(
                        f"{counter}. {row['Institution']} - {row['Programme_level']} - [{row['Programme_name']}]({row['Link']})")
                else:
                    st.write(f"{counter}. [{row['Institution']}]({row['Link']})")
                counter += 1

 
    st.write('---')

    components.html(
    """
    <a rel="license" href="http://creativecommons.org/licenses/by/4.0/"><img alt="Creative Commons Licence" style="border-width:0" 
    src="https://i.creativecommons.org/l/by/4.0/80x15.png" /></a><br />
    © 2022 All rights reserved. This website is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">Creative Commons Attribution 4.0 International License</a>.
    """
    )
