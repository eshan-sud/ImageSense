
# filename - src/script.py

# venv\scripts\activate
# pip install streamlit opencv-python numpy scikit-image scikit-learn matplotlib tensorflow
# pip freeze > src/requirements.txt
# streamlit run src/srcipt.py

import streamlit as st
import opencv3


def upload_image(): pass


st.title("ImageSense")
st.header("")
st.subheader("")

sidebar_elements = ""

sidebar = st.sidebar(sidebar_elements)


# st.file_uploader(<prompt>)
# st.image(uploaded_image)
# st.button(<text>)
# st.number_input(<prompt>, <min>, <max>)

# st.progress(<max>)
# st.spinner(<text>)
# st.success(<text>)
# st.error(<text>)
# st.warning(<text>)
# st.info(<text>)
# st.sidebar(<element(s)>)
# st.container()
