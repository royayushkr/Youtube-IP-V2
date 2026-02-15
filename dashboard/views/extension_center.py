import streamlit as st

from dashboard.extensions.registry import DEFAULT_EXTENSIONS, load_extensions, save_extensions


def render() -> None:
    st.title("Extension Center")
    st.write("Toggle modules in your dashboard suite.")

    current = load_extensions()
    updated = current.copy()

    for key in DEFAULT_EXTENSIONS:
        updated[key] = st.toggle(key, value=current.get(key, True), key=f"ext_{key}")

    c1, c2 = st.columns(2)
    if c1.button("Save Extension Settings", type="primary", use_container_width=True):
        save_extensions(updated)
        st.success("Saved. Reload sidebar selection if needed.")

    if c2.button("Reset to Defaults", use_container_width=True):
        save_extensions(DEFAULT_EXTENSIONS)
        st.success("Reset to defaults.")

    st.markdown("---")
    st.caption("Extensions are configured in `config/extensions.json`.")
