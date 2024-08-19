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
from streamlit_gsheets import GSheetsConnection
from events import evens_conferences


st.set_page_config(layout = "wide", 
                    page_title='Intelligence studies network',
                    page_icon="https://images.pexels.com/photos/315918/pexels-photo-315918.png",
                    initial_sidebar_state="auto") 

st.title("Intelligence studies network")
st.header('Resources on intelligence studies')
st.info('''
        This page lists institutions, academic programs, and  other resources on intelligence studies. 
        **Please note that this list may not be complete. If you have suggestions, please [get in touch](https://forms.gle/qjWwpysJpTWCJh1c7)!**
        ''')


image = 'https://images.pexels.com/photos/315918/pexels-photo-315918.png'

with st.sidebar:

    sidebar_content()


col1, col2 = st.columns([5,2])
with col1:
    conn = st.connection("gsheets", type=GSheetsConnection)
    df = conn.read(spreadsheet='https://docs.google.com/spreadsheets/d/1Xv8wo9nzFrcKfjDxPZz2WDPhi-2uXNvHMDOYMTJwZeE/edit#gid=0')

    countries = df['Country'].unique()
    types = df['Type'].unique()

    df = df[df['Status'] == 'Active']
    df = df.sort_values(by='Institution')
    df_plot = df.copy()

    def display_numbered_list(programs, column_name, show_country=False, show_programme_level=True):
        counter = 1
        for index, row in programs.iterrows():
            programme_name = row['Programme_name']
            programme_info = ""
            
            if pd.notna(programme_name):
                if show_programme_level and column_name == 'Academic programs':
                    programme_info = f"{row['Programme_level']}: [{programme_name}]({row['Link']}), *{row['Institution']}*, {row['Country']}"
                else:
                    programme_info = f"[{programme_name}]({row['Link']}), *{row['Institution']}*, {row['Country']}"
                if show_programme_level and column_name == 'Other resources':
                    programme_info = f"{row['Programme_level']}: [{programme_name}]({row['Link']}), *{row['Institution']}*, {row['Country']}"
            else:
                if show_programme_level and column_name == 'Other resources':
                    programme_info = f"{row['Programme_level']}: [{row['Institution']}]({row['Link']}), {row['Country']}"
                else:
                    programme_info = f"[{row['Institution']}]({row['Link']}), {row['Country']}"
            
            if show_country:
                programme_info += f", {row['Country']}"
            
            st.write(f"{counter}. {programme_info}")
            counter += 1

    for prog_type in types:
        type_programs = df[df['Type'] == prog_type]
        num_unique_countries = type_programs['Country'].nunique()

        if prog_type == 'Government institutions':
            countries = type_programs['Country'].unique()
            with st.expander(f"**{prog_type} ({len(type_programs)})**"):
                if num_unique_countries != 1:
                    country_counts = type_programs['Country'].value_counts().sort_values(ascending=False)
                    countries_sorted = country_counts.index.tolist()
                    country_counts_dict = {country: f"{country} ({count})" for country, count in country_counts.items()}
                    selected_countries = st.multiselect('Filter by country:', countries_sorted, format_func=lambda x: country_counts_dict[x])
                    
                    if selected_countries:
                        type_programs = type_programs[type_programs['Country'].isin(selected_countries)]
                        num_unique_countries = type_programs['Country'].nunique()
                        if num_unique_countries == 1:
                            selected_country_str = selected_countries[0].split(" (")[0]
                            st.write(f'**{len(type_programs)} {prog_type} found in {selected_country_str}**')
                        #else:
                            # st.write(f'**{len(type_programs)} {prog_type} found in {num_unique_countries} countries**')
                    else:
                        selected_countries = countries_sorted  # Show all countries if none is selected
                        type_programs = df[df['Type'] == prog_type]  # Show all programs initially

                if len(selected_countries) == 1:
                    country_programs = type_programs[type_programs['Country'] == selected_countries[0]]
                    # st.write(f'**{len(country_programs)} {prog_type} found in {selected_countries[0].split(" (")[0]}**')
                    display_numbered_list(country_programs, prog_type, show_country=False, show_programme_level=False)
                else:
                    st.write(f'**{len(type_programs)} {prog_type} found in {num_unique_countries} countries**')
                    for country in selected_countries:
                        country_programs = type_programs[type_programs['Country'] == country]
                        st.markdown(f'##### {country}')
                        display_numbered_list(country_programs, prog_type, show_country=False, show_programme_level=False)
        else:
            with st.expander(f"**{prog_type} ({len(type_programs)})**"):
                if prog_type == 'Academic programs':
                    country_counts = type_programs['Country'].value_counts().sort_values(ascending=False)
                    countries_sorted = country_counts.index.tolist()
                    country_counts_dict = {country: f"{country} ({count})" for country, count in country_counts.items()}
                    
                    # Filter by country
                    selected_country = st.multiselect('Filter by country:', countries_sorted, format_func=lambda x: country_counts_dict[x])
                    if selected_country:
                        type_programs_filtered = type_programs[type_programs['Country'].isin(selected_country)]
                        countries_to_display = selected_country
                    else:
                        type_programs_filtered = type_programs.copy()
                        countries_to_display = countries_sorted

                    # Filter by program type
                    programme_levels = type_programs_filtered['Sub_type'].unique()
                    selected_programme_level = st.multiselect("Filter by Programme Level:", programme_levels)
                    if selected_programme_level:
                        type_programs_filtered = type_programs_filtered[type_programs_filtered['Sub_type'].isin(selected_programme_level)]

                    num_filtered_countries = type_programs_filtered['Country'].nunique()

                    on = st.toggle('Display as barchart')
                    if on:
                        country_program_counts = type_programs_filtered.groupby(['Country', 'Sub_type']).size().reset_index(name='Count')
                        country_totals = country_program_counts.groupby('Country')['Count'].sum().reset_index(name='Total_Count')
                        sorted_countries = country_totals.sort_values(by='Total_Count', ascending=False)['Country'].tolist()

                        # Sort country_program_counts based on the total count within each country
                        country_program_counts = country_program_counts.merge(country_totals, on='Country')
                        country_program_counts = country_program_counts.sort_values(by=['Total_Count', 'Count'], ascending=[False, False])

                        sorted_countries_reverse = sorted_countries[::-1]  # Reversing the list

                        # Create the plot
                        fig = px.bar(country_program_counts, x='Count', y='Country', orientation='h', color='Sub_type',
                                    category_orders={"Sub_type": sorted(programme_levels)})

                        # Set the order of countries in the plot (reversed)
                        fig.update_layout(
                            title='Number of Academic Programs by Country',
                            xaxis_title='Number of Programs',
                            yaxis_title='Country',
                            yaxis={'categoryorder': 'array', 'categoryarray': sorted_countries_reverse}  # Set the reversed order
                        )
                        st.plotly_chart(fig)
                    else:
                        st.write('The chart is hidden')

                        if num_filtered_countries > 1:
                            st.write(f'**{len(type_programs_filtered)} {prog_type} found in {num_filtered_countries} countries**')
                        elif num_filtered_countries == 1:
                            selected_country_str = countries_to_display[0].split(" (")[0]
                            st.write(f'**{len(type_programs_filtered)} {prog_type} found in {selected_country_str}**')
                        else:
                            st.write(f'**No {prog_type} found for the selected filters**')

                        for country in countries_to_display:
                            country_programs = type_programs_filtered[type_programs_filtered['Country'] == country]
                            count_country_programs = len(country_programs)
                            country_display_name = f"{country} ({count_country_programs})"
                            if count_country_programs > 0:
                                st.markdown(f'### {country_display_name}')
                                display_numbered_list(country_programs, prog_type, show_country=False, show_programme_level=True)

                else:
                    if num_unique_countries != 1:
                        num_unique_countries = type_programs['Country'].nunique()
                        country_counts = type_programs['Country'].value_counts().sort_values(ascending=False)
                        countries_sorted = country_counts.index.tolist()
                        country_counts_dict = {country: f"{country} ({count})" for country, count in country_counts.items()}
                        selected_country = st.multiselect('Filter by country:', countries_sorted, format_func=lambda x: country_counts_dict[x])

                        if selected_country:
                            type_programs = type_programs[type_programs['Country'].isin(selected_country)]
                            num_unique_countries = type_programs['Country'].nunique()
                            if num_unique_countries == 1:
                                selected_country_str = selected_country[0].split(" (")[0]
                                st.write(f'**{len(type_programs)} {prog_type} found in {selected_country_str}**')
                            else:
                                st.write(f'**{len(type_programs)} {prog_type} found in {num_unique_countries} countries**')
                        else:
                            st.write(f'**{len(type_programs)} {prog_type} found in {num_unique_countries} countries**')

                    display_numbered_list(type_programs, prog_type, show_country=False if prog_type != 'Academic' else False)

    df_plot = df_plot[df_plot['Type'] != 'Government institutions']
    df_plot = df_plot.groupby(['Country', 'Type']).size().reset_index(name='Count')
    country_totals = df_plot.groupby('Country')['Count'].sum().reset_index(name='Total_Count')
    sorted_countries = country_totals.sort_values(by='Total_Count', ascending=False)['Country'].tolist()

    # Sort country_program_counts based on the total count within each country
    df_plot = df_plot.merge(country_totals, on='Country')
    df_plot = df_plot.sort_values(by=['Total_Count', 'Count'], ascending=[False, False])
    sorted_countries_reverse = sorted_countries[::-1]  # Reversing the list

    # Create the plot
    fig = px.bar(df_plot, x='Count', y='Country', orientation='h', color='Type',
                    category_orders={"Type": sorted(programme_levels)})

    # Set the order of countries in the plot (reversed)
    fig.update_layout(
        title='Number of Academic Institutions & Programs by Country',
        xaxis_title='Number of Institutions & Programs',
        yaxis_title='Country',
        yaxis={'categoryorder': 'array', 'categoryarray': sorted_countries_reverse}  # Set the reversed order
    )
    st.plotly_chart(fig) 

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

    with st.expander('Digest', expanded=True):
        st.write('See our dynamic [digest](https://intelligence.streamlit.app/Digest) for the latest updates on intelligence!')
st.write('---')

components.html(
"""
<a rel="license" href="http://creativecommons.org/licenses/by/4.0/"><img alt="Creative Commons Licence" style="border-width:0" 
src="https://i.creativecommons.org/l/by/4.0/80x15.png" /></a><br />
© 2024 Yusuf Ozkan. All rights reserved. This website is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">Creative Commons Attribution 4.0 International License</a>.
"""
)
