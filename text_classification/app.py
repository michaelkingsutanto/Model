#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pickle
import pandas as pd

# Load your model
with open('text_classification/model_classification.pkl', 'rb') as f:
    model = pickle.load(f)

# Title
st.title("Text Classification with Naive Bayes")

# User text input
user_input = st.text_area("Enter text here")

if st.button('Predict'):
    prediction = model.predict([user_input])[0]
    if prediction == 0:
        st.write("Input doesn't align with Omega community")
    else:
        st.write("Input aligns with Omega community")

