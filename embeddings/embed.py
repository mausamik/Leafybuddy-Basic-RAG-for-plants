from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from langchain_chroma import Chroma
import os 

load_dotenv()  # Load environment variables from .env file

COLLECTION_NAME = "plant_db"
PERSIST_DIRECTORY = "./chroma_langchain_db"


def get_embeddings():
    return HuggingFaceEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )


def embed_documents(documents):
    ## Keep a check to see if database already exists
    if os.path.exists(PERSIST_DIRECTORY):
        print("Database already exists. Loading existing database...")
        vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=get_embeddings(),
        )
    else:
        localembeddings = get_embeddings()
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=localembeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=PERSIST_DIRECTORY,
    )
    ## stored into vectordb
    return vectorstore




