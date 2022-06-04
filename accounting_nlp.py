import pandas as pd
import numpy as np
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# take input files
uploaded_files = st.file_uploader("Choose file(s) to upload", accept_multiple_files=True)
if uploaded_files:
     texts = []
     file_names = []
     for uploaded_file in uploaded_files:
          bytes_data = uploaded_file.read()
          texts.append(bytes_data)
          file_names.append(uploaded_file.name)
          #st.write("filename:", uploaded_file.name)
          #st.write(bytes_data)


     # Initialize an instance of tf-idf Vectorizer
     tfidf_vectorizer = TfidfVectorizer()

     # Generate the tf-idf vectors for the corpus
     tfidf_matrix = tfidf_vectorizer.fit_transform(texts)

     # compute and print the cosine similarity matrix
     cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

     similarity = []
     for i in range(0, len(texts)):
       similarity.append(["{:.0%}".format(str(round(j, 2))) for j in cosine_sim[i]])
     #"{:.0%}".format(
     
     df_labels = []
     for i in range(0, len(texts)):
       df_labels.append(file_names[i].split(".txt")[0])

     df = pd.DataFrame(similarity)
     df.columns = df_labels

     # Change the row indexes
     df.index = df_labels
     st.table(df)
     
     @st.cache
     def convert_df(df):
          return df.to_csv().encode('utf-8')

     csv = convert_df(df)

     st.download_button(
          label="Download data as CSV",
          data=csv,
          file_name='document_similarities.csv',
          mime='text/csv',
      )
