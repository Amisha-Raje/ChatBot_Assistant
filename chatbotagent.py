import os
from langchain_openai import OpenAI
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_cohere import CohereRerank
import re
from langchain_google_genai import ChatGoogleGenerativeAI

# from embeddingManager import embedding


load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["COHERE_API_KEY"] = os.getenv("COHERE_API_KEY")


class chatbotAgent:
    def __init__(self,vectordb,temperature = 0):
        self.vectordb = vectordb
        # self.llm = OpenAI(temperature = temperature, verbose = True)
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro",
            # google_api_key = GOOGLE_API_KEY,
            temperature=0,
            max_tokens=None,
            timeout=None,
        )
        self.llm1 = Ollama(model="phi3",verbose = True)
        self.rag_chain = None
        self.query_transformer_chain = None
        self.retriever_chain = None


    def setupBot(self):
        retriever = self.vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 6})
        compressor = CohereRerank()
        # compressor = FlashrankRerank()
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=retriever
        )
        
        
        # def format_docs(docs):
        #     return "\n\n".join(doc.page_content for doc in docs)
        def format_docs(docs):
            formatted_content = []
            for doc in docs:
                if(doc.metadata.get("relevance_score")>0.5):
                    content = doc.page_content.replace("\\n", "\n")
                    # Remove unnecessary consecutive newlines
                    content = " ".join([line.strip() for line in content.split("\n") if line.strip()])
                    source = doc.metadata.get("source", "Unknown Source")
                    # print(source)
                    formatted_content.append(f"Content:{content}\n\nSource:{source}")
            return "\n\n".join(formatted_content)
            
        def format_docs1(docs):
            formatted_content = []
            contents = []
            for doc in docs:
                content = doc.page_content.replace("\\n", "\n")
                # Remove unnecessary consecutive newlines
                content = " ".join([line.strip() for line in content.split("\n") if line.strip()])
                source = [doc.metadata.get("source", "Unknown Source"),doc.metadata.get("relevance_score",0.1)]
                # print(source)
                formatted_content.append(source)
                contents.append(content)
            print("\n\n".join(contents))
            return formatted_content

        self.retriever_chain = compression_retriever | format_docs1
    

        template_query_transformer = """ You a question re-writer that converts an input question to a better version that is optimized for vectorstore retrieval.
            Look at the input and try to reason about the underlying semantic intent / meaning and higlighting important keywords.

            
            Here is the initial question: \n{question} \n 
            
            Formulate and give only improved question.

        """


        query_transformer_prompt = PromptTemplate.from_template(template_query_transformer)
        
        # template_final_ans = """You are an assistant for question-answering tasks and you dont have any knowledge other than context that i will provide. Use only the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know, don't try to make up an answer and keep it very short.
        #     keep the answer concise.Always say "thanks for asking!" at the end of the answer.
        #     {context}

        #     Question:{question}

        #     Helpful Answer:
        # """
        # template_final_ans = """You are an AI assistant for question-answering tasks and you don't have any knowledge other than the context that I provide. Use only the following pieces of retrieved context only to answer the question. If the answer is not present in the provided context, just say "I don't know". Keep the answer very short and concise. Always say "thanks for asking!" at the end of the answer.
        #     context:[
        #     {context}
        #     ]

        #     Question: {question}

        #     Helpful Answer:
        # """

        # template_final_ans = """You are an AI assistant for question-answering tasks and you don't have any knowledge other than the context that I provide. Use only the following pieces of retrieved context to answer the question. If the answer is not present in the provided context, just say "I don't know". Keep the answer very short, point-wise, and informative. Always say "thanks for asking!" at the end of the answer.
        #     context:[
        #     {context}
        #     ]

        #     Question: {question}

        #     Helpful Answer:
        # """
        template_final_ans = """Use only the following pieces of context below to answer the question at the end. If the answer is not present in the provided context, just say "I don't know" don't try to make up an answer. Keep the answer pointwise and short (at max 10 sentences). Always say "thanks for asking!" at the end of the answer.
            context:[
            {context}
            ]

            Question: {question}

            Helpful Answer:
        """

        
        template_final_ans1 = """You are an AI assistant trained for question-answering tasks. You have access to contexts I provided below.
            Please use this information to generate helpful answers to the given questions. If you're unable to provide a definitive answer, simply state that you don't know.
            Cite the sources along with your response whhich i gave you in context .Always say "thanks for asking!" at the end of the answer.

            context:
            {context}

            Question:
            {question}

            Helpful Answer:

            Sources:   
        """

        rag_prompt = PromptTemplate.from_template(template_final_ans)

        # rag_prompt = ChatPromptTemplate.from_messages(
        #     [
        #         ("system","You are an assistant for question-answering tasks and you dont have any knowledge other than context that i will provide. Use only the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know, don't try to make up an answer and keep it very short.keep the answer concise.Always say thanks for asking! at the end of the answer."),
        #         # ("context","{context}"),
        #         ("human","{context}")
        #     ]
        # )
        # query_transformer_prompt = ChatPromptTemplate.from_messages(
        #     [
        #         ("system","You a question re-writer that converts an input question to a better version that is optimized for vectorstore retrieval.Look at the input and try to reason about the underlying semantic intent / meaning and higlighting important keywords and output transformed question only."),
        #         ("human","{context}")
        #     ]
        # )
        
        # memory = ConversationBufferWindowMemory( k=1, return_messages=True)

        # chain = create_stuff_documents_chain(
        #     llm = self.llm,
        #     prompt = rag_prompt,
        #     output_parser = StrOutputParser()
        # )
        # self.query_transformer_chain = create_stuff_documents_chain(
        #     llm = self.llm1,
        #     prompt = query_transformer_prompt,
        #     # output_parser = StrOutputParser()
        # )


        # self.query_transformer_chain = (
        #     {"question":RunnablePassthrough()}
        #     |query_transformer_prompt
        #     |self.llm 
        #     |StrOutputParser()
        # )
        self.query_transformer_chain1 = (
            {"question":RunnablePassthrough()}
            |query_transformer_prompt
            |self.llm1
            |StrOutputParser()
        )
        
        self.rag_chain = (
            {"context":compression_retriever | format_docs, "question":RunnablePassthrough()}
            | rag_prompt
            | self.llm
            | StrOutputParser()
        )
        # self.rag_chain = create_retrieval_chain((retriever|format_docs),chain)

    def ask_question(self,query):
        # transformed_query = self.query_transformer_chain.invoke({"question": query})
        # transformed_query1 = self.query_transformer_chain1.invoke({"question": query})

        # print(transformed_query)
        # print(transformed_query1)
        # result = self.rag_chain.invoke(query)
        # print(result1)
        sources = []
        # sources = [context.replace('Markdown_Files/', '') for context in result1 if context != "Unknown Source"]
        # for context in result:
        #     if(context[0]!= "Unknown Source"  and context[1]>0.5):
        #         a = context[0].replace(".md",'')
        #         pattern = r"final_files_copy/batch_\d+/"
        #         new_path = re.sub(pattern, '', a)
        #         b = new_path.replace('MarkDown_Files/', '')
        #         c = "https://dev-outline.sprinklr.com/doc/"+b
        #         print(c)
        #         sources.append(c)
        # print(result1)
        result = self.rag_chain.invoke(query)

        final_source = list(set(sources))
        # result = "hehehehe"
        return [result,final_source]
        # return transformed_query
        # return "fef"
        
        

# model_host = os.getenv('MODEL_HOST', 'localhost')
# model_port = os.getenv('MODEL_PORT', '11434')
# endpoint_url = f"http://{model_host}:{model_port}"

# embedding_function = OllamaEmbeddings(model = "nomic-embed-text",base_url = endpoint_url)
embedding_function = OllamaEmbeddings(model = "nomic-embed-text")
db = Chroma(persist_directory = "./db_final_test", embedding_function=embedding_function)
chatbot = chatbotAgent(db)
chatbot.setupBot()