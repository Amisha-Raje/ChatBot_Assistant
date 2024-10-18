from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma



embedding_function = OllamaEmbeddings(model = "nomic-embed-text")
db = Chroma(persist_directory = "./db_final_test_1", embedding_function=embedding_function)

docs = db.similarity_search("what is knn?")
print(docs[0])