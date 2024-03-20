import streamlit.components.v1 as components
import datetime
import streamlit as st

current_year = datetime.datetime.now().year  
cite_today = datetime.date.today().strftime("%d %B %Y")

def display_custom_license():
    st.write(f'**Cite this page:** Ozkan, Yusuf A. ‘*Intelligence Studies Network*’, Created 1 June 2020, Accessed {cite_today}. https://intelligence.streamlit.app/.')
    st.write(f'© {current_year} Yusuf Ozkan. All rights reserved. This website is licensed under a [Creative Commons Attribution 4.0 International License](http://creativecommons.org/licenses/by/4.0/).')
    components.html(
    f"""
    <br><a rel="license" href="http://creativecommons.org/licenses/by/4.0/"><img alt="Creative Commons Licence" style="border-width:0" 
    src="https://i.creativecommons.org/l/by/4.0/80x15.png" /></a><br />
    © {current_year} Yusuf Ozkan. All rights reserved. This website is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">Creative Commons Attribution 4.0 International License</a>.
    </br>
    <br>
    <strong>Cite this page:</strong> Ozkan, Yusuf A. ‘<em>Intelligence Studies Network</em>’, Created 1 June 2020, Accessed {cite_today}. <a href="https://intelligence.streamlit.app/">https://intelligence.streamlit.app/</a>.
    </br>

    <p xmlns:cc="http://creativecommons.org/ns#" xmlns:dct="http://purl.org/dc/terms/"><a property="dct:title" rel="cc:attributionURL" href="https://intelligence.streamlit.app/">'Intelligence Studies Network'</a> by <span property="cc:attributionName">Yusuf Ozkan</span> is licensed under <a href="http://creativecommons.org/licenses/by/4.0/?ref=chooser-v1" target="_blank" rel="license noopener noreferrer" style="display:inline-block;">CC BY 4.0<img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/cc.svg?ref=chooser-v1"><img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/by.svg?ref=chooser-v1"></a></p>
    """
    )
