import os

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
load_dotenv()  # Load environment variables from .env file


def format_docs(docs):
    """Combines the text of all retrieved chunks into a single string for the prompt context"""
    return "\n\n".join(doc.page_content for doc in docs)


def generate_content(userquery, plant_retriever):
    """
    Function to generate content using the HuggingFace API.
    Args:
        userquery: The input query for content generation.
    Returns:
        The generated content as a string.
    """
    ##model_id = os.getenv("", "Qwen/Qwen2.5-7B-Instruct")

    model = ChatOllama(
        model ="qwen3:0.6b",
        temperature= 0.5, 
        num_predict = 1000 ,
        disable_streaming = True
    )

   ## model = ChatHuggingFace(llm = llm)

    system_prompt = (
        "You are an expert botanical and home plant assistant for LeafyHome India.\n"
        "Answer the user's question using ONLY the provided context below. "
        "If you do not know the answer based on the context, say 'I cannot find that in my library.' "
        "Do not extrapolate or invent facts outside this text.\n\n"
        "Context:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    chain = ({"context" : plant_retriever | format_docs, 
              "input" : RunnablePassthrough()}
              | prompt | model | StrOutputParser() 
              )

    final_output = chain.invoke(userquery)
    return final_output