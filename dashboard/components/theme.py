import streamlit as st


def inject_theme() -> None:
    st.markdown(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&display=swap');

:root {
  --bg-0: #0a111b;
  --bg-1: #111c2b;
  --bg-2: #16283c;
  --text-0: #eef4ff;
  --text-1: #b8c6db;
  --brand-0: #3dd598;
  --brand-1: #49a6ff;
  --line: rgba(153, 184, 210, 0.25);
}

html, body, [class*="css"] {
  font-family: 'Space Grotesk', sans-serif;
}

.stApp {
  background: radial-gradient(circle at 20% 10%, #1e3550 0%, #0a111b 45%), linear-gradient(180deg, var(--bg-1), var(--bg-0));
}

section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #121f31 0%, #0d1725 100%);
  border-right: 1px solid var(--line);
}

.hero-wrap {
  border: 1px solid var(--line);
  border-radius: 16px;
  padding: 24px;
  background: linear-gradient(140deg, rgba(73,166,255,0.2), rgba(61,213,152,0.1));
  margin-bottom: 14px;
}

.hero-title {
  font-size: 34px;
  font-weight: 700;
  color: var(--text-0);
  letter-spacing: 0.3px;
}

.hero-sub {
  margin-top: 8px;
  font-size: 15px;
  color: var(--text-1);
}

.tool-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 10px;
  margin-top: 8px;
  margin-bottom: 8px;
}

.tool-card {
  border: 1px solid var(--line);
  border-radius: 14px;
  padding: 14px;
  background: rgba(13, 24, 39, 0.72);
}

.tool-card h4 {
  margin: 0 0 6px;
  color: var(--text-0);
}

.tool-card p {
  margin: 0;
  color: var(--text-1);
  font-size: 14px;
  line-height: 1.35;
}

[data-testid="stMetric"] {
  border: 1px solid var(--line);
  border-radius: 12px;
  padding: 8px;
  background: rgba(13, 24, 39, 0.65);
}
</style>
        """,
        unsafe_allow_html=True,
    )
