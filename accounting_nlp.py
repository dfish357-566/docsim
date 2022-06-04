import pandas as pd
import numpy as np
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


#from collections import namedtuple
#import altair as alt


# take input files
uploaded_files = st.file_uploader("Choose a CSV file", accept_multiple_files=True)
for uploaded_file in uploaded_files:
     bytes_data = uploaded_file.read()
     st.write("filename:", uploaded_file.name)
     st.write(bytes_data)


print(bytes_data)
'''
def open_text(file_name):
  text_file = open(file_name, "r")
  data = text_file.read()
  text_file.close()
  return data
'''

#file_names = ["apple_2019.txt", "apple_2020.txt", "microsoft_2019.txt", "microsoft_2020.txt"]

#file_names = 

texts = []
for i in file_names:
  texts.append(open_text(i))



# Initialize an instance of tf-idf Vectorizer
tfidf_vectorizer = TfidfVectorizer()

# Generate the tf-idf vectors for the corpus
tfidf_matrix = tfidf_vectorizer.fit_transform(texts)

# compute and print the cosine similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
print(cosine_sim)

for i in range(0, 4):
  print(file_names[i], [str(round(j, 2)) for j in cosine_sim[i]])

similarity = []
for i in range(0, 4):
  similarity.append([str(round(j, 2)) for j in cosine_sim[i]])

df_labels = []
for i in range(0, 4):
  df_labels.append(file_names[i].split(".txt")[0])

df = pd.DataFrame(similarity)
df.columns = df_labels
  
# Change the row indexes
df.index = df_labels
print(df.to_string())
