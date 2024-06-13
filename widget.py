import streamlit as st

st.title("Streamlit Widgets Example")

# Slider
value = st.slider("Select a value", 0, 100, 50)
st.write("Selected value:", value)

# Selectbox
option = st.selectbox("Select an option", ["Option 1", "Option 2", "Option 3"])
st.write("Selected option:", option)
