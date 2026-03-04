# pyright: reportMissingImports=false
# pyright: reportMissingModuleSource=false

from src.helper import load_pdf_files,filter_to_minimal_docs,text_split,download_embeddings

extracted_pdf_files = load_pdf_files("data")
minimal_docs = filter_to_minimal_docs(extracted_pdf_files)
text_chunks = text_split(minimal_docs)
embedding = download_embeddings()

#################################################################################################################

from dotenv import load_dotenv
import os
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

#################################################################################################################

from pinecone import Pinecone         #Imports the Pinecone client class so you can talk to Pinecone from Python.
pinecone_api_key = PINECONE_API_KEY  

pc = Pinecone(api_key=pinecone_api_key)  #Creates a connection object to Pinecone using your API key.(logged-in Pinecone session)

# We are creating index inside the Pinecode.
# Just like Table is SQL database --> Index in vector databases

from pinecone import ServerlessSpec#Imports ServerlessSpec, which tells Pinecone where and how to create the index (cloud + region).
index_name = "medical-chatbot"

if not pc.has_index(index_name):
    pc.create_index(
        name = index_name,
        dimension=384,  # Dimension of the embeddings
        metric= "cosine",  # Cosine similarity
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(index_name)

#################################################################################################################

from langchain_pinecone import PineconeVectorStore

docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    embedding=embedding,
    index_name=index_name
)


