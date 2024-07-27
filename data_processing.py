import re
from transformers import BertTokenizer, TFBertModel
import tensorflow as tf
import numpy as np
import faiss

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased') #Helps convert raw text into a format the model can understand
model = TFBertModel.from_pretrained('bert-base-uncased') #Helps Extract contexual relationships among tokens - > Transormer layers

def clean_text(document_text):
    cleaned_text = document_text.lower() #For uniformity
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text) #single space uniformity
    cleaned_text = re.sub(r'[^\w\s£]', '', cleaned_text) #Removing punctuation expect the '£' sign -> Required for the objective
    return cleaned_text


def embed_text_content(sentences):
    inputs = tokenizer(sentences, return_tensors='tf', padding=True, truncation=True)
    outputs = model(inputs) 
    return tf.reduce_mean(outputs.last_hidden_state, axis=1).numpy()

def embed_user_query(query):
    '''This function cleans and returns the user input query'''
    cleaned_query = clean_text(query)
    return embed_text_content(cleaned_query)


def get_index_for_text_embeddings(document_embeddings):
    #Building Index - FAISS (FaceBook AI Similarity Search), since FAISS is optimized to handle high-dimensional vectors of BERT
    document_embeddings_array = np.array(document_embeddings)
    index = faiss.IndexFlatL2(document_embeddings_array.shape[1])
    index.add(document_embeddings_array)
    return index

def get_top_k_most_similar_documents(index, query, k=5):
    query_embedding = embed_user_query(query)
    distance, indices = index.search(query_embedding, k)
    return indices


