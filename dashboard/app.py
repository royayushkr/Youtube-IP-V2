import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dashboard.components.sidebar import render_sidebar
from dashboard.components.theme import inject_theme
from dashboard.extensions.registry import get_navigation_items
from dashboard.views import channel_analysis, extension_center, home, recommendations, ytuber


st.set_page_config(page_title="YouTube IP Dashboard", page_icon="📺", layout="wide")
inject_theme()

PAGES = {
    "Home": home.render,
    "Channel Analysis": channel_analysis.render,
    "Recommendations": recommendations.render,
    "Ytuber": ytuber.render,
    "Extension Center": extension_center.render,
    "Deploy Notes": None,
}

nav_items = [p for p in get_navigation_items() if p in PAGES]
page = render_sidebar(nav_items)

if page == "Deploy Notes":
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
        python3 -m streamlit run dashboard/app.py
        ```

        ### Streamlit Cloud
        1. Push to GitHub.
        2. Create app with entrypoint `dashboard/app.py`.
        3. Add required secrets.
        """
    )
else:
    render_fn = PAGES.get(page)
    if render_fn is None:
        st.error("Selected page is not configured.")
    else:
        render_fn()
