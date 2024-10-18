# import os
# from dotenv import load_dotenv
# from langchain_community.document_loaders import UnstructuredMarkdownLoader
# from langchain_community.document_loaders import DirectoryLoader
# from langchain_text_splitters import MarkdownHeaderTextSplitter
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.embeddings import OllamaEmbeddings
# from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
# from langchain_community.vectorstores import Chroma
# import concurrent.futures
# import shutil
# import logging
# import time

# load_dotenv()

# os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2")
# os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # Set the logging level for 'unstructured' to WARNING to suppress INFO messages
# logging.getLogger("unstructured").setLevel(logging.WARNING)
# logging.getLogger("chromadb.api.segment").setLevel(logging.WARNING)


# class DocumentManager:
#     def __init__(self,directory_path,glob_pattern = "./*.md"):
#         self.directory_path = directory_path
#         self.glob_pattern = glob_pattern
#         self.documents = []
#         self.all_sections = []
#         self.final_chunks = []

#     def load_document(self):
#         UnstructuredMarkdownLoader_kwargs = {"mode":"single"}
#         loader = DirectoryLoader(self.directory_path,glob = self.glob_pattern,show_progress = True, loader_cls = UnstructuredMarkdownLoader , loader_kwargs = UnstructuredMarkdownLoader_kwargs) 
#         self.documents = loader.load()

#     def split_document(self):
#         header_to_split_on = [("#","Header 1"),("##","Header 2"),("###","Header 3"),("####","Header 4"),]
#         makrdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=header_to_split_on,strip_headers = False, return_each_line = True)
#         for doc in self.documents:
#             sections = makrdown_splitter.split_text(doc.page_content)
#             self.all_sections.extend(sections)
#         chunk_size = 1000
#         chunk_overlap = 200
#         text_splitter = RecursiveCharacterTextSplitter(chunk_size = chunk_size, chunk_overlap = chunk_overlap)
#         self.final_chunks = text_splitter.split_documents(self.documents)


# class embedding_database_manager:
#     def __init__(self,final_chunks,persist_directory = "./db2"):
#         self.final_chunks = final_chunks
#         self.persist_directory = persist_directory
#         self.vectordb  = None

#     def embedding_and_store(self):
#         embedding = OllamaEmbeddings(model="nomic-embed-text")

#         # Define a function to embed and store each chunk
#         def embed_and_store(chunk):
#             vectordb = Chroma.from_documents(documents=[chunk], embedding=embedding, persist_directory=self.persist_directory, collection_name="db2",tenant="default_tenant")
#             vectordb.persist()

#         # Use ThreadPoolExecutor to parallelize the embedding and storing process
#         with concurrent.futures.ThreadPoolExecutor(max_workers = 1) as executor:
#             futures = [executor.submit(embed_and_store, chunk) for chunk in self.final_chunks]
#             for future in concurrent.futures.as_completed(futures):
#                 future.result()  # Wait for all futures to complete


# # vectordb = embedding.vectordb

# time1 = time.perf_counter()
# doc = DocumentManager("./final_files")
# doc.load_document()
# doc.split_document()
# print(len(doc.final_chunks))

# # final_doc = doc.final_chunks


# # # # embedding and storing the chunks in vector database
# embedding = embedding_database_manager(doc.final_chunks)
# embedding.embedding_and_store()
# time2 = time.perf_counter()
# print(f"time taken{time2-time1}")

        


import os
from dotenv import load_dotenv
from langchain_community.document_loaders import UnstructuredMarkdownLoader, DirectoryLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
import concurrent.futures
import logging
import time

load_dotenv()

# os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2")
# os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set the logging level for 'unstructured' to WARNING to suppress INFO messages
logging.getLogger("unstructured").setLevel(logging.WARNING)
logging.getLogger("chromadb.api.segment").setLevel(logging.WARNING)


class DocumentManager:
    def __init__(self,directory_path,glob_pattern = "./*.md"):
        self.directory_path = directory_path
        self.glob_pattern = glob_pattern
        self.documents = []
        self.all_sections = []
        self.final_chunks = []

    def load_document(self):
        UnstructuredMarkdownLoader_kwargs = {"mode":"single"}
        loader = DirectoryLoader(self.directory_path,glob = self.glob_pattern,show_progress = True, loader_cls = UnstructuredMarkdownLoader , loader_kwargs = UnstructuredMarkdownLoader_kwargs) 
        self.documents = loader.load()

    def split_document(self):
        header_to_split_on = [("#","Header 1"),("##","Header 2"),("###","Header 3"),("####","Header 4"),]
        makrdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=header_to_split_on,strip_headers = False, return_each_line = True)
        for doc in self.documents:
            sections = makrdown_splitter.split_text(doc.page_content)
            self.all_sections.extend(sections)
        chunk_size = 1000
        chunk_overlap = 200
        text_splitter = RecursiveCharacterTextSplitter(chunk_size = chunk_size, chunk_overlap = chunk_overlap)
        self.final_chunks = text_splitter.split_documents(self.documents)


# class EmbeddingDatabaseManager:
#     def __init__(self, final_chunks, persist_directory="./db_mark_test"):
#         self.final_chunks = final_chunks
#         self.persist_directory = persist_directory
#         self.vectordb = None

#     def embedding_and_store(self):
#         embedding = OllamaEmbeddings(model="nomic-embed-text")

#         # Initialize Chroma once
#         vectordb = Chroma(embedding_function=embedding, persist_directory=self.persist_directory, collection_name="db_mark_tes")

#         # Define a function to embed and store each chunk
#         def embed_and_store(chunk):
#             print(chunk)
#             print("\n")
#             vectordb.add_documents([chunk])

#         # Use ThreadPoolExecutor to parallelize the embedding and storing process
#         with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
#             futures = [executor.submit(embed_and_store, chunk) for chunk in self.final_chunks]
#             for future in concurrent.futures.as_completed(futures):
#                 try:
#                     future.result()  # Wait for all futures to complete
#                     print("hehe")
#                 except Exception as e:
#                     logger.error(f"Error processing chunk: {e}")
        
#         vectordb.persist()
class EmbeddingDatabaseManager:
    def __init__(self, final_chunks, persist_directory="./db_mark_test1"):
        self.final_chunks = final_chunks
        self.persist_directory = persist_directory
        self.vectordb = None

    def embedding_and_store(self):
        embedding = OllamaEmbeddings(model="nomic-embed-text")

        # Initialize Chroma once
        vectordb = Chroma(embedding_function=embedding, persist_directory=self.persist_directory, collection_name="db_mark_test1")

        # Define a function to embed and store each chunk
        def embed_and_store(chunk):
            # print(chunk)
            # print("\n")
            vectordb.add_documents([chunk])

        # Use ThreadPoolExecutor to parallelize the embedding and storing process
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(embed_and_store, chunk) for chunk in self.final_chunks]
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()  # Wait for all futures to complete
                    print("hehe")
                except Exception as e:
                    logger.error(f"Error processing chunk: {e}")
        
        vectordb.persist()




if __name__ == "__main__":
    time1 = time.perf_counter()
    doc = DocumentManager("./MarkDown_Files")
    doc.load_document()
    doc.split_document()
    print(f"Number of final chunks: {len(doc.final_chunks)}")

    # Embedding and storing the chunks in the vector database
    embedding = EmbeddingDatabaseManager(doc.final_chunks)
    embedding.embedding_and_store()
    time2 = time.perf_counter()
    print(f"Time taken: {time2 - time1} seconds")
