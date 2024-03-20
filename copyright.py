import streamlit.components.v1 as components
import datetime
import streamlit as st

current_year = datetime.datetime.now().year  
cite_today = datetime.date.today().strftime("%d %B %Y")

def display_custom_license():
    st.write(f'**Cite this page:** Ozkan, Yusuf A. ‘*Intelligence Studies Network*’, Created 1 June 2020, Accessed {cite_today}. https://intelligence.streamlit.app/.')
    components.html(
    f"""
    <br><a rel="license" href="http://creativecommons.org/licenses/by/4.0/"><img alt="Creative Commons Licence" style="border-width:0" 
    src="https://i.creativecommons.org/l/by/4.0/80x15.png" /></a><br />
    © {current_year} Yusuf Ozkan. All rights reserved. This website is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">Creative Commons Attribution 4.0 International License</a>.
    </br>
    <br>
    <strong>Cite this page:</strong> Ozkan, Yusuf A. ‘<em>Intelligence Studies Network</em>’, Created 1 June 2020, Accessed {cite_today}. <a href="https://intelligence.streamlit.app/">https://intelligence.streamlit.app/</a>.
    </br>
    """
    )
