import streamlit as st

def apply_styles():
    st.markdown("""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@400;600;700&display=swap');
            * {
                font-family: 'Cormorant Garamond', serif;
                font-weight: 700;
                font-size: 110%;
            }
        </style>
    """, unsafe_allow_html=True)