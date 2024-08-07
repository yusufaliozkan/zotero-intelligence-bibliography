import streamlit as st
import streamlit.components.v1 as components
from copyright import display_custom_license, cc_by_licence_image

def sidebar_content():
    image = 'https://images.pexels.com/photos/315918/pexels-photo-315918.png'
    
    with st.sidebar:
        st.image(image, width=150)
        st.logo(
            image=image,
            icon_image=image,
            link='https://intelligence.streamlit.app/'
            )
        st.sidebar.markdown("# Intelligence studies network")
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
        with st.expander('Issues'):
            st.warning('''
            Links to PhD theses catalouged by the British EThOS may not be working due to the [cyber incident at the British Library](https://www.bl.uk/cyber-incident/).
            ''')
        with st.expander('Contact us'):
            st.write('''
            Join our [mailing list](https://groups.google.com/g/intelligence-studies-network) to receive updates about the website.

            Contact [me](https://kcsi.uk/members/yusuf-ozkan).

            Report your technical issues or requests [here](https://github.com/YusufAliOzkan/zotero-intelligence-bibliography/issues).            
            ''')
        st.write('Check the digest [here](https://intelligence.streamlit.app/Digest)')
        st.toast('Join our [mailing list](https://groups.google.com/g/intelligence-studies-network) to receive updates.')

