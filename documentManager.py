import os
from dotenv import load_dotenv
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
import shutil
import logging

load_dotenv()

# os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2")
# os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set the logging level for 'unstructured' to WARNING to suppress INFO messages
logging.getLogger("unstructured").setLevel(logging.WARNING)


class DocumentManager:
    def __init__(self,directory_path,glob_pattern = "./*.md"):
        self.directory_path = directory_path
        self.glob_pattern = glob_pattern
        self.documents = []
        self.all_sections = []
        self.final_chunks = []
    def doc_batch(self,batch_size = 60):
        all_files = [f for f in os.listdir(self.directory_path) if os.path.isfile(os.path.join(self.directory_path, f))]
    
        # Create batches of files
        for i in range(0, len(all_files), batch_size):
            batch_files = all_files[i:i + batch_size]
            batch_folder = os.path.join(self.directory_path, f'batch_{i // batch_size + 1}')
        
        # Create a new batch folder 
        os.makedirs(batch_folder, exist_ok=True)           
        
        # Move files to the new batch folder
        for file in batch_files:
            shutil.move(os.path.join(self.directory_path, file), os.path.join(batch_folder, file))
        # print(f'Moved {len(batch_files)} files to {batch_folder}')


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
        chunk_size = 500
        chunk_overlap = 100
        text_splitter = RecursiveCharacterTextSplitter(chunk_size = chunk_size, chunk_overlap = chunk_overlap)
        self.final_chunks = text_splitter.split_documents(self.documents)

# doc = DocumentManager("./final_files_copy/batch_1")
# doc = DocumentManager("./final_files_copy/batch_2")
# doc = DocumentManager("./final_files_copy/batch_3")
# doc = DocumentManager("./final_files_copy/batch_4")
# doc = DocumentManager("./final_files_copy/batch_5")
# doc = DocumentManager("./final_files_copy/batch_6")
# doc = DocumentManager("./final_files_copy/batch_7")
# doc = DocumentManager("./final_files_copy/batch_8")  
# doc = DocumentManager("./final_files_copy/batch_9")
# doc = DocumentManager("./final_files_copy/batch_10")
# doc = DocumentManager("./final_files_copy/batch_11")
# doc = DocumentManager("./final_files_copy/batch_12")
# doc = DocumentManager("./final_files_copy/batch_13")
# doc = DocumentManager("./final_files_copy/batch_14")
# doc = DocumentManager("./final_files_copy/batch_15")
# doc = DocumentManager("./final_files_copy/batch_16")
doc = DocumentManager("./MarkDown_Files")
doc.load_document()
doc.split_document()
print(doc.final_chunks[0])

# final_doc = doc.final_chunks

        
