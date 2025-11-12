# streamlit_app.py
import streamlit as st
from pathlib import Path
from engine.ui import run_app


if __name__ == '__main__':
    # Optional: configure page here
    st.set_page_config(page_title='Hermis Prism', page_icon='💎', layout='wide')
    run_app()

#goto: https://hermis.streamlit.app/