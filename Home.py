import streamlit as st
import pandas as pd

# home_page = st.Page('Home.py', title='Affiliation finder')

home = st.Page('Home_page.py', title='Home')

intelligence_history = st.Page('pages/1_Intelligence history.py', title='Intelligence history')
intelligence_studies = st.Page('pages/2_Intelligence studies.py', title='Intelligence studies')
intelligence_analysis = st.Page('pages/3_Intelligence analysis.py', title='Intelligence analysis')
intelligence_organisations = st.Page('pages/4_Intelligence organisations.py', title='Intelligence organisations')
intelligence_failures = st.Page('pages/5_Intelligence failures.py', title='Intelligence failures')
intelligence_oversight = st.Page('pages/6_Intelligence oversight and ethics.py', title='Intelligence oversight and ethics')
intelligence_collection = st.Page('pages/7_Intelligence collection.py', title='Intelligence collection')
counterintelligence = st.Page('pages/8_Counterintelligence.py')
covert_action = st.Page('pages/9_Covert action.py')
intelligence_cybersphere = st.Page('pages/10_Intelligence and cybersphere.py')
global_intelligence = st.Page('pages/11_Global intelligence.py')
special_collections = st.Page('pages/13_Special collections.py')

events = st.Page('pages/14_Events.py')
digest = st.Page('pages/15_Digest.py')
institutions = st.Page('pages/16_Institutions.py')
item_monitoring = st.Page('pages/17_Item monitoring.py')

pg = st.navigation(
    {
        'Home':[home],
        'Collections':[
            intelligence_history, 
            intelligence_studies,
            intelligence_analysis,
            intelligence_organisations,
            intelligence_failures,
            intelligence_oversight,
            intelligence_collection,
            counterintelligence,
            covert_action,
            intelligence_cybersphere,
            global_intelligence,
            special_collections
            ],
        'Other resources':[
            events,
            digest,
            institutions,
            item_monitoring
        ]
    }
)
    
pg.run()