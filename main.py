import data_loading as dl
import data_processing as dp
import faiss
import numpy as np
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer
import tensorflow as tf


#Loading the documents
documents_list = dl.get_documents()

#Cleaning the documents
cleaned_documents = [dp.clean_text(document) for document in documents_list]

#Embedding the documents
document_embeddings = dp.embed_text_content(cleaned_documents)

#Getting Index for semantic searching
index = dp.get_index_for_text_embeddings(document_embeddings)

#Handling text Generation using an LLM -> GPT-2

model = 'openai-community/gpt2'
gpt2_tokenizer = GPT2Tokenizer.from_pretrained(model)
gpt2_model = TFGPT2LMHeadModel.from_pretrained(model)

# gpt2_tokenizer = AutoTokenizer.from_pretrained(model)
# gpt2_model = AutoModelForCausalLM.from_pretrained(model)

if gpt2_tokenizer.pad_token is None:
    gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token

def generate_text(prompt, max_length = 1000):
    inputs = gpt2_tokenizer.encode(prompt, return_tensors='tf', padding=True, truncation=True)
    # print(f'Inputs : {inputs}')
    # print(type(inputs))
    
    input_ids = inputs
    if len(input_ids.shape) == 1:
        input_ids = tf.expand_dims(input_ids, 0)
        
    outputs = gpt2_model.generate(input_ids=input_ids, max_length=max_length, num_return_sequences=1)
    return gpt2_tokenizer.decode(outputs[0], skip_special_tokens=True)


def rag_system(user_query, index):
    #Getting top matching documents
    indices = dp.get_top_k_most_similar_documents(index=index, query=user_query, k=5)
    relevant_documents_list = [cleaned_documents[i] for i in indices[0] if i != -1]
    combined_prompt = " ".join(relevant_documents_list) + " " + user_query
    model_response = generate_text(combined_prompt, 1000)
    return model_response

# user_query = 'How much is for Supreme of Chicken and Cream Sauce'
user_query = input('Enter your question!::\n')
response = rag_system(user_query=user_query, index=index)
print(response)





