import streamlit.components.v1 as components
import datetime
import streamlit as st


def _today_strings():
    today = datetime.date.today()
    return today.year, today.strftime("%d %B %Y")


def display_custom_license():
    current_year, cite_today = _today_strings()
    components.html(
        f"""
        <br><a rel="license" href="http://creativecommons.org/licenses/by/4.0/"><img alt="Creative Commons Licence" style="border-width:0"
        src="https://i.creativecommons.org/l/by/4.0/80x15.png" /></a><br />
        © {current_year} IntelArchive. All rights reserved. This website is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">Creative Commons Attribution 4.0 International License</a>.
        </br>
        <br>
        <strong>Cite this page:</strong> IntelArchive ‘<em>Intelligence Studies Network</em>’, Created 1 June 2020, Accessed {cite_today}. <a href="https://intelarchive.app/">https://intelarchive.app/</a>.
        </br>
        """
    )


def cc_by_licence_image():
    current_year, _ = _today_strings()
    components.html(
        f"""
        <br><a rel="license" href="http://creativecommons.org/licenses/by/4.0/"><img alt="Creative Commons Licence" style="border-width:0"
        src="https://i.creativecommons.org/l/by/4.0/80x15.png" /></a><br />
        © {current_year} IntelArchive. All rights reserved. This website is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">Creative Commons Attribution 4.0 International License</a>.
        </br>
        <br>
        """
    )