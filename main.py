#create a streamlit app that takes email content and returns a prediction of whether the email is spam or not
import streamlit as st
import pandas as pd
import numpy as np
import vertexai
from vertexai.language_models import TextGenerationModel
import os



service_account_creds={
  "type": "service_account",
  "project_id": "orbital-nova-365616",
  "private_key_id": "8dabc753c10d76fd5ef52a384ebc9c49fbb219a7",
  "private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQCsYqxFlSvsQoTV\nTK8Gntc8sr+tzgYtaZ5DYrPYpfINNac3rZcY/rbBmDeZL5Vaad0jdcinkwMt8hax\naJpK3QTAotiixSgSMhuojMv/Voq9y8oCOz9twSQFxlbuFKDI3NHdz//RxH9t1ktP\nMZ5chvjdUNWp7gheYKmPdXGbEB7+J7IxYqnQnQfJd/0RFvr55EdJFxlgxYxilZod\nw+59LjBLJ6SmTyph+26lO86YD6Gon7e5ywdUryUTu5eOOhGsL8R1+TUNaMadGnI2\n0lhSAu+tIWTERXy3lyPGIjAkRcsk26lUI55XWZIcp4HpjGFCtMsht+WAWzQ0IGur\nnshAtzY/AgMBAAECggEAVP9KJf0JbTLXQDzRSBhl1D3mmRIupZGgQWWXe8lG8hB0\nZNWo3OAqyIX/U/7MS4pLSUK4LzgjpzHo0ozbFvKndzxUSN2hhmdCj/bsVvga7L5g\n2nzQJ5PF6TsEfduZ87A9onr+jjWvBz9UXrX+eWzrpGRJFcKSScarlFq6K1Tlkza8\nTZIPAx0PhoFKpGpYGh9BI0NGJ2Rj5KI/ggXpJZBV9MpLQJJ95jW1nbcvxPeC0EDJ\nG2wUe1zfkgC7XwclOXcopyv1s014r9betJjsVudR6A2ZmY0uy105Yxp8Aq/HgNUo\nkUgrRIW+pZil6494RLgWyt6O0zCO+ectGYJOUh+2UQKBgQDU6c65e7wySHyCVO5J\nPXFv7hQas8qujC8k7vI7j1hgH3zv1V/gi1x0rpXjqd9FaJD3/R4y46bXOE/rmnFY\n3+K4wxCi/wfiVnyYywjAjUVnKwxnlRY8DnT2Azpzm00EQgvH2dKQUeU1UYXOCFg/\nRPAKWIwa2vChs+1YSKa8QvBIRQKBgQDPRUcOeXJkgQ4af2c6gJviaTB1F6gl39u4\n0i5I+xnmoRKhVhwoCVOa+udoPlog1cqjb+IVhPhPQqUxOPUJeTmejyR81JGAttzb\n0VqUJDjgXvl0NxGm+vRZtiKzcmUeyVxIq3HVLnfOrF3+siZ11snHl2SIg1r+69SG\nBC0ENU/WswKBgCkXFnZlkuSBaMYebf5khUf8wpSErlERbOfL1aP4/i1O5mXC4EYN\nn0JFj1foqNZUX+X8S5+2n7RZgdMOc2Gbk73kp4lNftAHsPQr+vxuRz6p6DL1at0c\nTSZNaTJ2fl6cAHoJZ8HCjGHF4fTpV16eh1T30dI8/sO2Xc/z9XnP0V+hAoGBAKOj\ngUb3MQ4SKa1X2osAkJG/FGfNho5oX1NadPpfbIJ9tnSzxEUDoAMiheHyIDv+Lh68\nrVTqBxSkfEERQ7v3/jqlrlvzgvCI7nOJpKaCtnjF4tISzBf110agmmLsNNOKjBY/\nbP6u0h3UTd1QjzbHncS7TL7OXeAXFlG8KEJDSMYrAoGANlLcbNTDHAMFwNwHSebP\nR52AAqVtT1QC9suhEzs+UeFKhWupONEwH52bqE4Lq/R7Y062cfbrAvBexGm17r9R\n2G6S61WYMzM2e/fYkW1/jkpWaF22jiDdeNbOZ18HiJ8jfJGt+wRUUbbKmjWY5syM\njuN3G21rD/JOXuMwfsBaT/k=\n-----END PRIVATE KEY-----\n",
  "client_email": "genai-srmist-23@orbital-nova-365616.iam.gserviceaccount.com",
  "client_id": "115199199161409966222",
  "auth_uri": "https://accounts.google.com/o/oauth2/auth",
  "token_uri": "https://oauth2.googleapis.com/token",
  "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
  "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/genai-srmist-23%40orbital-nova-365616.iam.gserviceaccount.com",
  "universe_domain": "googleapis.com"
}

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = './privcreds.json'
vertexai.init(project="orbital-nova-365616", location="us-central1")
parameters = {
    "candidate_count": 1,
    "max_output_tokens": 200,
    "temperature": 0.2,
    "top_p": 0.8,
    "top_k": 40
}
model = TextGenerationModel.from_pretrained("text-bison")

st.title('Spam Email Classifier')
st.write('This app predicts whether an email is spam or not')
st.write('Please enter your mail details below')

mail_sender=st.text_input('Enter the sender of the mail')
mail_subject=st.text_input('Enter the subject of the mail')
mail=st.text_area('Paste your email here')

request_string = "provided the body of the mail , what are the chances of this mail being a spam  and if so why ? \n\n\n" + "sender mail = "+ mail_sender + "\n\n mail subject = " + mail_subject + "\n\n mail = "  + mail



response = model.predict(
    request_string,**parameters)
st.write(response.text)