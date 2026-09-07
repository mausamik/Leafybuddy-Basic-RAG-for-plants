from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

def split_documents_recursively(documents):
    textsplitter = RecursiveCharacterTextSplitter(
        chunk_size = 500, 
        chunk_overlap  = 30, 
    )

    split_docs = textsplitter.split_documents(documents)
    return split_docs


def split_docs_semantic(documents):

    localembeddings = HuggingFaceEmbeddings(
       model = "sentence-transformers/all-MiniLM-L6-v2",
       model_kwargs = {"device": "cpu"},
    )
    semantic_splitter = SemanticChunker(
        embeddings = localembeddings,
        min_chunk_size = 500,
        
    )
    return semantic_splitter.split_documents(documents)


