import os
import re
from dotenv import load_dotenv
from typing import List
from typing_extensions import TypedDict
from langchain_openai import OpenAI
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_cohere import CohereRerank
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from openAiWrapper import ChatIntuition


load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["COHERE_API_KEY"] = os.getenv("COHERE_API_KEY")

class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""

    binary_score: str = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )

class GradeAnswer(BaseModel):
    """Binary score to assess answer addresses question."""

    binary_score: str = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )
class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        content: list of documents
        sources: list of sources
        flag: to check infinite loop
        last5chats: chat history upto last 5 chats
    """

    question: str
    flag: int
    last5Chats:list[str]
    generation: str
    content: str
    sources: list[str]





class chatbotAgent:
    def __init__(self,vectordb,temperature = 0):
        self.vectordb = vectordb
        self.llm = ChatOpenAI(temperature = temperature, verbose = True)
        self.llm1 = ChatOpenAI(temperature = temperature,verbose = True)
        self.llm2 = ChatOpenAI(temperature = temperature,verbose = True)
        self.rag_chain = None
        self.contextualise_query_chain = None
        self.retriever_chain = None
        self.hallucination_grader = None
        self.answer_grader = None
        self.app = None

    def graph(self):
        ########### nodes ########### 
        def contextualise(state):
            """
            contextualise query

            Args:
                state (dict): The current graph state

            Returns:
                state (dict): contextualises query with the last 5 chat history
            """
            print("---GENERATE---")
            question = state["question"]
            content = state["content"]
            sources = state["sources"]
            flag = state["flag"]
            last5Chats = state["last5Chats"]
            generation = state["generation"]
            print(f"--last5chats---\n{last5Chats}\n\n")
            # RAG generation
            question = self.contextualise_query_chain.invoke({"last5Chats":last5Chats,"question":question}).content
            print(f"--Contextualised query---\n{question}\n\n")
            return {"content": content,"sources":sources, "question": question,"flag":flag,"last5Chats":last5Chats,"generation":generation}
      

        def retrieve(state):
            """
            Retrieve documents

            Args:
                state (dict): The current graph state

            Returns:
                state (dict): New key added to state, documents, that contains retrieved documents
            """
            print("---RETRIEVE---")
            question = state["question"]
            flag = state["flag"]
            last5Chats = state["last5Chats"]
            generation = state["generation"]
            # Retrieval
            documents = self.retriever_chain.invoke(question)
            content = documents[0]
            sources = documents[1]
            return {"content": content,"sources":sources, "question": question,"flag":flag,"last5Chats":last5Chats,"generation":generation}
        
        def generate(state):
            """
            Generate answer

            Args:
                state (dict): The current graph state

            Returns:
                state (dict): New key added to state, generation, that contains LLM generation
            """
            print("---GENERATE---")
            question = state["question"]
            content = state["content"]
            sources = state["sources"]
            flag = state["flag"]
            flag+=1
            last5Chats = state["last5Chats"]
            # RAG generation
            generation = self.rag_chain.invoke({"context": content, "question": question})
            print(f"--content---\n{content}\n\n")
            print(f"--generation---\n{generation}\n\n")
           
            return {"content": content,"sources":sources, "question": question,"flag":flag,"last5Chats":last5Chats,"generation":generation}
        
        def ask_user(state):
            """
            Ask query 

            Arg:
                state (dict): The current graph state
            Returns:
                state (dict): updated qustion key with re-phrased question
            """
            print("---Asking user query")
            question = state["question"]
            content = state["content"]
            sources = state["sources"]
            generation = state["generation"]
            flag = state["flag"]
            last5Chats = state["last5Chats"]
            generation += "\nThis is all I know please ask more specific query!"
            return {"content": content,"sources":sources, "question": question,"flag":flag,"last5Chats":last5Chats,"generation":generation}
            # to complete 
        ########### nodes ########### 

        ########### edge ########### 
        def grade_generation_v_documents_and_question(state):
            """
            Determines whether the generation is grounded in the document and answers question.

            Args:
                state (dict): The current graph state

            Returns:
                str: Decision for next node to call
            """

            print("---CHECK HALLUCINATIONS---")
            question = state["question"]
            content = state["content"]
            sources = state["sources"]
            generation = state["generation"]
            last5Chats = state["last5Chats"]
            flag = state["flag"]
            grade = "yes"
            if(flag<2):
                score = self.hallucination_grader.invoke(
                    {"content": content, "generation": generation}
                )
                grade = score.binary_score
            # Check hallucination
            if grade == "yes":
                print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
                # Check question-answering
                print("---GRADE GENERATION vs QUESTION---")
                score = self.answer_grader.invoke({"question": question, "generation": generation})
                grade = score.binary_score
                if grade == "yes":
                    print("---DECISION: GENERATION ADDRESSES QUESTION---")
                    return "useful"
                else:
                    print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
                    return "not useful"
            else:
                print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
                return "not supported"
        ########### edge ########### 


        ########### connecting edges to nodes ########### 
        workflow = StateGraph(GraphState)
        # workflow = self.workflow

        # defining nodes
        workflow.add_node("retrieve",retrieve)
        workflow.add_node("generate",generate)
        workflow.add_node("ask_user",ask_user)
        workflow.add_node("contextualise",contextualise)
       

        #building graph
        workflow.set_entry_point("contextualise")
        workflow.add_edge("contextualise","retrieve")
        workflow.add_edge("retrieve","generate")
        workflow.add_conditional_edges(
            "generate",
            grade_generation_v_documents_and_question,
            {
                "not supported": "generate",
                "useful": END,
                "not useful": "ask_user",
            },
        )
        workflow.add_edge("ask_user",END)
        ########### connecting edges to nodes ########### 

        ########### compiling graph ########### 
        self.app = workflow.compile()
        ########### compiling graph ########### 


    def setupBot(self):

        ########### retriever ########### 
        retriever = self.vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 25})
        compressor = CohereRerank()
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=retriever
        )
        ########### retriever ########### 

        ########### llm setup ########### 
        structured_llm_grader_hallucination =self.llm1.with_structured_output(GradeHallucinations)
        structured_llm_grader_answer =self.llm2.with_structured_output(GradeAnswer)
        ########### llm setup ########### 
                
        ########### formatting docs ########### 
        def format_docs(docs):
            formatted_content = []
            for doc in docs:
                if(doc.metadata.get("relevance_score")>0.3):
                    content = doc.page_content.replace("\\n", "\n")
                    content = " ".join([line.strip() for line in content.split("\n") if line.strip()])
                    source = doc.metadata.get("source", "Unknown Source")
                    formatted_content.append(f"Content:{content}\n\nSource:{source}")
            return "\n\n".join(formatted_content)
            
        def format_docs1(docs):
            contents = []
            sources = []
            for doc in docs:
                if(doc.metadata.get("relevance_score")>0.5 and doc.metadata.get("source")!="Unknown Source"):
                    content = doc.page_content.replace("\\n", "\n")
                    content = " ".join([line.strip() for line in content.split("\n") if line.strip()])
                    contents.append(content)
                    source = [doc.metadata.get("source", "Unknown Source"),doc.metadata.get("relevance_score",0.1)]
                    sources.append(source)
            return ["\n\n".join(contents),sources]
        ########### formatting docs ########### 



        ########### prompt messages  ########### 
        template_final_ans = """Use only the following pieces of context below to answer the question at the end. If the answer is not present in the provided context, just say "I don't know" don't try to make up an answer. Keep the answer pointwise and short (at max 10 sentences). Always say "thanks for asking!" at the end of the answer.
            context:[
            {context}
            ]

            Question: {question}

            Helpful Answer:
        """
        answer_generation = """You are an assistant for question-answering tasks. Use ONLY the following pieces of retrieved context to generate answer to the question.
            If the answer is NOT present in the provided context, say that you don't know. 
            Keep the answer POINTWISE and SHORT (at max 10 sentences). Always say "thanks for asking!" at the end of the answer."""
        
        system_hallucination = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n 
            Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""
        
        system_answer = """You are a grader assessing whether an answer addresses / resolves a question \n 
            Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question."""
        
        contextual_query_transformer = """ Given a chat history and the latest user question which might reference context in the chat history, 
            formulate a standalone question which can be understood without the chat history. Do NOT answer the question, 
            just reformulate it if needed and otherwise return it as is."""
        ########### prompt messages ########### 

        ########### prompt ########### 
        # query_transformer_prompt = PromptTemplate.from_template(template_query_transformer)
        contextualise_query_prompt = ChatPromptTemplate.from_messages(
            [
                ("system",contextual_query_transformer),
                ("human", "chat history:\n{last5Chats} \n\n latest user question:\n{question}")
            ]
        )
        answer_generation_prompt = ChatPromptTemplate.from_messages(
            [
                ("system",answer_generation),
                ("human","retrieved context:\n{context} \n\n question:{question}"),
            ]
        )
        # rag_prompt = PromptTemplate.from_template(template_final_ans)
        hallucination_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_hallucination),
                ("human","Set of facts: \n {content} \n\n LLM generation: {generation}"),
            ]
        )
        answer_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_answer),
                ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
            ]
        )
        ########### prompt ########### 

        ########### chains ########### 
        self.retriever_chain = compression_retriever | format_docs1
        self.contextualise_query_chain = contextualise_query_prompt | self.llm
        # self.rag_chain = (
        #     {"last5Chats":RunnablePassthrough, "context":RunnablePassthrough(), "question":RunnablePassthrough()}
        #     | rag_prompt
        #     | self.llm
        #     | StrOutputParser()
        # )
        self.rag_chain = answer_generation_prompt | self.llm | StrOutputParser()
        self.hallucination_grader = hallucination_prompt | structured_llm_grader_hallucination
        self.answer_grader = answer_prompt | structured_llm_grader_answer
        ########### chains ########### 

    def ask_question(self,query,last5Chats):
        inputs = {
            "question":query,
            "flag":0,
            "last5Chats":last5Chats
        }
        for output in self.app.stream(inputs):
            for key, value in output.items():
                print(f"Node '{key}':")
            print("\n---\n")            
        sources = []
        result = value["sources"]
        generation = value["generation"]
        for context in result:
            if(context[0]!= "Unknown Source"  and context[1]>0.3):
                a = context[0].replace(".md",'')
                pattern = r"final_files_copy/batch_\d+/"
                new_path = re.sub(pattern, '', a)
                b = new_path.replace('MarkDown_Files/', '')
                c = "https://dev-outline.sprinklr.com/doc/"+b
                sources.append(c)
        final_source = list(set(sources))
        # return [generation,final_source]
        return [generation]

        
# model_host = os.getenv('MODEL_HOST', 'localhost')
# model_port = os.getenv('MODEL_PORT', '11434')
# endpoint_url = f"http://{model_host}:{model_port}"

# embedding_function = OllamaEmbeddings(model = "nomic-embed-text",base_url = endpoint_url)
endpoint_url = "http://localhost:11434"
embedding_function = OllamaEmbeddings(model="nomic-embed-text", base_url=endpoint_url)

# embedding_function = OllamaEmbeddings(model = "nomic-embed-text")
# db = Chroma(persist_directory = "./src/db_final_test", embedding_function=embedding_function)
db = Chroma(persist_directory = "./db_final_test", embedding_function=embedding_function)
chatbot = chatbotAgent(db)
chatbot.setupBot()
chatbot.graph()
