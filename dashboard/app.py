import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dashboard.components.sidebar import render_sidebar
from dashboard.views import channel_analysis, recommendations, ytuber


st.set_page_config(page_title="YouTube IP Dashboard", page_icon="📺", layout="wide")


page = render_sidebar()

if page == "Channel Analysis":
    channel_analysis.render()
elif page == "Recommendations":
    recommendations.render()
elif page == "Ytuber":
    ytuber.render()
else:
    st.title("Deploy Notes")
    st.markdown(
        """
        ### Environment Variables
        - `YOUTUBE_API_KEY` for channel stats pull
        - `GEMINI_API_KEY` for titles/descriptions/scripts/thumbnails
        - `OPENAI_API_KEY` optional fallback for thumbnail generation

        ### Local Run
        ```bash
        source .venv/bin/activate
        streamlit run dashboard/app.py
        ```

        ### Streamlit Cloud
        1. Push this repo to GitHub.
        2. Create a new Streamlit app with entrypoint `dashboard/app.py`.
        3. Add secrets for required API keys.
        """
    )
