import streamlit as st
import numpy as np
import pandas as pd
import nltk
import re
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem.snowball import SnowballStemmer
from collections import Counter
import joblib

# Load data
df = pd.read_csv("Book_summary.csv")

# Reset index and rename columns
df = df.reset_index()
df = df.rename(columns={"index": "id"})

# Select a subset of the data (optional)
df1 = df.head(500)

# Fill NaN values
df1['summary'] = df1['summary'].fillna('')

corpus = df1['summary']

from nltk.stem.snowball import SnowballStemmer
Stemmer = SnowballStemmer('english')

def tokenize_and_stem(text):
    tokens = [word for sentence in sent_tokenize(text) for word in word_tokenize(sentence)]
    filtered_tokens = []
    for token in tokens:
        if re.search('[a-zA-z]', token):
            filtered_tokens.append(token)
    stems = [Stemmer.stem(token) for token in filtered_tokens]
    return stems

def tokenize_only(text):
    # Your tokenize_only function here
    tokens = [word.lower() for sentence in nltk.sentence_tokenizer(text) for word in nltk.word_tokenizer(sentence)]
    filtered_tokens = []
    for token in tokens:
        if re.search('[a-zA-z]',token):
            filtered_tokens.append(token)
    return filtered_toekns

# Create TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=200000, min_df=0.2,
                                   stop_words='english', use_idf=True,
                                   tokenizer=tokenize_and_stem, ngram_range=(1, 3))

# Fit the TF-IDF vectorizer
tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)

# Load SentenceTransformer model
model = SentenceTransformer('sentence-transformers/msmarco-MiniLM-L6-cos-v5')

# Encode the text summaries
corpus_embeddings = model.encode(corpus)

# Perform K-Means clustering
num_clusters = 50
km = KMeans(n_clusters=num_clusters)
km.fit(tfidf_matrix)
clusters = km.labels_.tolist()

# Save the K-Means model
joblib.dump(km, 'doc_cluster.pkl')

# Load the K-Means model
km = joblib.load('doc_cluster.pkl')

# Create a Streamlit app
st.title('TF-IDF Clustering')

# Display clusters and associated information
for cluster_id in range(num_clusters):
    st.header(f'Cluster {cluster_id}')
    st.write("Number of documents in this cluster:", clusters.count(cluster_id))
    
    # Display top terms for the cluster
    order_centroids = km.cluster_centers_.argsort()[:, ::-1]
    top_terms = [tfidf_vectorizer.get_feature_names_out()[ind] for ind in order_centroids[cluster_id, :6]]
    st.write("Top terms for this cluster:", ', '.join(top_terms))
    
    # Display titles and authors of documents in the cluster
    cluster_indices = [i for i, c in enumerate(clusters) if c == cluster_id]
    cluster_documents = df1.iloc[cluster_indices]
    st.write("Documents in this cluster:")
    st.write(cluster_documents[['title', 'author']])
    st.write("---")

