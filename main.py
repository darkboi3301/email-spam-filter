#create a streamlit app that takes email content and returns a prediction of whether the email is spam or not
import streamlit as st
import pandas as pd
import numpy as np
import vertexai
from vertexai.language_models import TextGenerationModel
import os

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = './privcreds.json'
vertexai.init(project="orbital-nova-365616", location="us-central1")
parameters = {
    "candidate_count": 1,
    "max_output_tokens": 1024,
    "temperature": 0.2,
    "top_p": 0.8,
    "top_k": 40
}
model = TextGenerationModel.from_pretrained("text-bison")

st.title('Spam Email Classifier')
st.write('This app predicts whether an email is spam or not')
st.write('Please enter your email below')

mail=st.text_input('Paste your email here')

request_string = "provoded the body of the mail , what are the chances of this mail being a spam  and if so why ? \n\n\n" + mail



response = model.predict(
    request_string,**parameters)
st.write(response.text)