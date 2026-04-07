import streamlit as st
import streamlit.components.v1 as components
from copyright import display_custom_license, cc_by_licence_image
from streamlit_theme import st_theme


def sidebar_content():
    image = 'https://images.pexels.com/photos/315918/pexels-photo-315918.png'
    
    with st.sidebar:
        # with open('images/02_icon/IntelArchive_Digital_Icon_Colour-Positive.svg', 'r') as file:
        #     svg_content_icon = file.read()
        # with open('images/01_logo/IntelArchive_Digital_Logo_Colour-Positive.svg', 'r') as file:
        #     svg_content_logo = file.read()


        theme = st_theme(key='sidebar')
        # Set the image path based on the theme
        if theme and theme.get('base') == 'dark':
            image_path_logo = 'images/01_logo/IntelArchive_Digital_Logo_Colour-Negative.svg'
            image_path_icon = 'images/02_icon/IntelArchive_Digital_Icon_Colour-Negative.svg'
        else:
            image_path_logo = 'images/01_logo/IntelArchive_Digital_Logo_Colour-Positive.svg'
            image_path_icon = 'images/02_icon/IntelArchive_Digital_Icon_Colour-Positive.svg'

        # Read and display the SVG image
        with open(image_path_logo, 'r') as file:
            svg_content_logo = file.read()
            st.image(svg_content_logo, width=150)  # Adjust the width as needed
        with open(image_path_icon, 'r') as file:
            svg_content_icon = file.read()

        # st.image(svg_content_logo, width=150)
        st.logo(
            image=svg_content_icon,
            icon_image=svg_content_icon,
            link='https://intelligence.streamlit.app/'
            )
        st.sidebar.markdown("# IntelArchive")
        with st.expander('About'):
            st.write('''This website lists secondary sources on intelligence studies and intelligence history.
            The sources are originally listed in the [Intelligence bibliography Zotero library](https://www.zotero.org/groups/2514686/intelligence_bibliography).
            This website uses [Zotero API](https://github.com/urschrei/pyzotero) to connect the *Intelligence bibliography Zotero group library*.
            To see more details about the sources, please vfisit the group library [here](https://www.zotero.org/groups/2514686/intelligence_bibliography/library). 
            Citation data for journal articles comes from [OpenAlex](https://openalex.org/).
            ''')
            st.write(''''
                     This website was built and the library is curated by [Yusuf Ozkan](https://www.kcl.ac.uk/people/yusuf-ali-ozkan) | 
                     [Twitter/X](https://twitter.com/yaliozkan) | [LinkedIn](https://www.linkedin.com/in/yusuf-ali-ozkan/) | 
                     [ORCID](https://orcid.org/0000-0002-3098-275X) | [GitHub](https://github.com/YusufAliOzkan) | [Linktree](https://linktr.ee/yusufozkan)
                     ''')
            display_custom_license()
        with st.expander('Source code'):
            st.info('''
            Source code of this app is available [here](https://github.com/YusufAliOzkan/zotero-intelligence-bibliography).
            ''')
        with st.expander('Disclaimer'):
            st.warning('''
            This website and the Intelligence bibliography Zotero group library do not list all the sources on intelligence studies. 
            The list is created based on the creator's subjective views.
            ''')
        with st.expander('Sponsors'): 
            st.markdown('''
            Proudly sponsored by the [King's Centre for the Study of Intelligence](https://kcsi.uk/)
            ''')
        with st.expander('Contact us'):
            st.write('''
            Join our [mailing list](https://groups.google.com/g/intelarchive) to receive updates about the website.

            Contact [me](https://kcsi.uk/members/yusuf-ozkan).

            Report your technical issues or requests [here](https://github.com/YusufAliOzkan/zotero-intelligence-bibliography/issues).            
            ''')
        st.write('Check the digest [here](https://intelligence.streamlit.app/Digest)')
        st.toast('Join our [mailing list](https://groups.google.com/g/intelarchive) to receive updates.')


def set_page_config():
    st.set_page_config(
        layout="wide",
        page_title="IntelArchive",
        page_icon="https://raw.githubusercontent.com/yusufaliozkan/clone-zotero-intelligence-bibliography/181b55d8cbe066fee0074cbbd9e0e6bfdfbed570/images/02_icon/IntelArchive_Digital_Icon_Colour-Negative.svg",
        initial_sidebar_state="auto"
    )

def set_page_config_centered():
    st.set_page_config(
        layout="centered",
        page_title="IntelArchive",
        page_icon="https://raw.githubusercontent.com/yusufaliozkan/clone-zotero-intelligence-bibliography/181b55d8cbe066fee0074cbbd9e0e6bfdfbed570/images/02_icon/IntelArchive_Digital_Icon_Colour-Negative.svg",
        initial_sidebar_state="auto"
    )