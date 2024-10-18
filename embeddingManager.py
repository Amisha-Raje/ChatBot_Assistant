from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from documentManager import doc
import time
import os
class embedding_database_manager:
    def __init__(self,final_chunks,persist_directory = "./db_final_test"):
        self.final_chunks = final_chunks
        self.persist_directory = persist_directory
        self.vectordb  = None

    def embedding_and_store(self):
        model_host = os.getenv('MODEL_HOST', 'localhost')
        model_port = os.getenv('MODEL_PORT', '11434')
        endpoint_url = f"http://{model_host}:{model_port}"
        embedding = OllamaEmbeddings(model = "nomic-embed-text",base_url=endpoint_url)
        # embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
        self.vectordb = Chroma.from_documents(documents = self.final_chunks,embedding = embedding,persist_directory = self.persist_directory)
        self.vectordb.persist()

# # embedding and storing the chunks in vector database
time1 = time.perf_counter()
embedding = embedding_database_manager(doc.final_chunks)
embedding.embedding_and_store()
time2 = time.perf_counter()
print(f"completed in {time2-time1} sec")

# vectordb = embedding.vectordb
