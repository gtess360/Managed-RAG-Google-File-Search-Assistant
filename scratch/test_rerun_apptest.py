from streamlit.testing.v1 import AppTest

code = """
import streamlit as st

if "c" not in st.session_state:
    st.session_state["c"] = 0

sidebar_val = st.sidebar.write(f"val: {st.session_state['c']}")

if st.button("inc"):
    st.session_state["c"] += 1
    st.rerun()
"""

at = AppTest.from_string(code).run()
print("Initial sidebar:", [m.value for m in at.sidebar.markdown])
at.button[0].click().run()
print("After click sidebar:", [m.value for m in at.sidebar.markdown])
print("Session state c:", at.session_state["c"])

