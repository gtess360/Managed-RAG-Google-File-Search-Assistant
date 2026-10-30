import streamlit as st
from streamlit.testing.v1 import AppTest

code = """
import streamlit as st
with st.container(border=True):
    st.markdown('<div class="chat-bubble-assistant"><b>Assistant</b></div>', unsafe_allow_html=True)
    st.markdown("Here is the answer with **markdown**.")
    with st.expander("Sources (1)"):
        st.write("**1. test_doc.md** (Page 1)")
        st.caption("Chunk ID: chunk_1")
        st.write('"Sample passage from document"')
"""
at = AppTest.from_string(code).run()
print("Expander labels:", [e.label for e in at.expander])
print("Markdown values:", [m.value for m in at.markdown])
print("Caption values:", [c.value for c in at.caption])

