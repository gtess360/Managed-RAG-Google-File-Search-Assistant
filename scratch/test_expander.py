import streamlit as st
from streamlit.testing.v1 import AppTest

code = """
import streamlit as st
with st.expander("Sources (1)"):
    st.write("doc.md")
    st.caption("chunk_1")
"""
at = AppTest.from_string(code).run()
print("expander labels:", [e.label for e in at.expander])
print("markdown values:", [m.value for m in at.markdown])
print("caption values:", [c.value for c in at.caption])

