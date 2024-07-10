import streamlit as st
from langchain_community.vectorstores import FAISS
# from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.chains import RetrievalQA

from langchain_community.document_transformers import (
    LongContextReorder,
)
from langchain.chains import LLMChain, StuffDocumentsChain
from groq import Groq
from langchain_groq import ChatGroq
import warnings
warnings.filterwarnings("ignore")
GROQ_API_KEY=st.secrets["GROQ_API_KEY"]

model_name="sentence-transformers/all-roberta-large-v1"
model_name="sentence-transformers/all-MiniLM-L6-v2"
min_emb = HuggingFaceEmbeddings(model_name=model_name)

# vectordb = Chroma(collection_name="base", embedding_function=min_emb, persist_directory="chromastore")

faissdb = FAISS.load_local("faissindex", embeddings=min_emb, allow_dangerous_deserialization=True)
retriever  = faissdb.as_retriever(search_kwargs={'k': 10}, return_source_documents=True, verbose=True)

model = "llama3-70b-8192"

# response = chain.run(input_documents=reordered_docs, query=query)
# print(response)

class Chat():
    def __init__(self, model):
        # Override prompts
        document_prompt = PromptTemplate(
            input_variables=["page_content"], template="{page_content}"
        )

        document_variable_name = "context"
        stuff_prompt_override = """
            INSTRUCTIONS 1.0:
                - You are a helpful assistant who answers questions based on the given context.
                - Just answer as you know the information. You can mention the book name.
                - Do not mention that you are giving answer based on the context provided.
                - Your task is to verify the question. If the question asked has anything to do with the context then only answer the question.
            Here is the context in triple quotes:
            '''{context}'''
            Here is the question in backticks:
            `{query}`

            Elaborate on all the points in the context
            """

        prompt = PromptTemplate(
            template=stuff_prompt_override, input_variables=["context", "query"]
        )
        llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name=model, temperature=1)
        # Instantiate the chain
        llm_chain = LLMChain(llm=llm, prompt=prompt)
        self.chain = StuffDocumentsChain(
            llm_chain=llm_chain,
            document_prompt=document_prompt,
            document_variable_name=document_variable_name,
        )
        self.docs = None

    def process_query(self, query):
        # Get relevant documents ordered by relevance score
        docs = retriever.invoke(query)
        reordering = LongContextReorder()
        self.docs = reordering.transform_documents(docs)[:5]
        # print(self.docs[0])
        # breakpoint()
        # print(len(self.docs[0]["page_content"]))

    def chat(self, query):
        self.process_query(query)
        try:    
            response = self.chain.run(input_documents=self.docs, query=query)
        except:
            return str("")
        return str(response)

# test
# x = Chat(model)
# query = input("> ")
# print(x.chat(query))
