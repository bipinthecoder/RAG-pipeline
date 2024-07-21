import os
import configuration as cfg


data_directory = cfg.document_directory_path


def get_documents():
    '''This function will load the documents and return the list of documents'''
    documents = []
    for file_name in os.listdir(data_directory):
        if file_name.endswith('.txt'):
            with open(os.path.join(data_directory, file_name), 'r', encoding='utf-8') as file:
                documents.append(file.read())
    
    return documents