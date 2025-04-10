import streamlit as st
from platforms.bluesky import Bluesky_Analyser

st.set_page_config(page_title="Social Media Analytic", layout="wide")

if 'selected_platform' not in st.session_state:
    st.title("Social Media Analytics Dashboard")
    st.markdown("""
    ### Disclaimer
        This tool uses pre trained AI models for its analysis which may contain some level of algorithmic bias
                
        It is recommended that the tool should be used as a support tool and not a decision making tool
                """)
    col1, col2  = st.columns(2)
    with col1:
        if st.button("Analyse a Bluesky profile"):
            st.session_state.selected_platform = "bluesky"
            st.rerun()
   # with col2:
   #     if st.button("Analyse a reddit profile"):
   #         st.session_state.selected_platform = "reddit"
   #         st.rerun()
    
    st.stop() 

if st.session_state.selected_platform == "bluesky":
    analyser = Bluesky_Analyser()
    analyser.run()


if st.button("Analyse Another Platform"):
    del st.session_state.selected_platform
    st.rerun()