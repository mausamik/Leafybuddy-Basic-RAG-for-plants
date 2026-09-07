from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, UnstructuredWordDocumentLoader

docs_path = Path(__file__).resolve().parents[1] / "docs"


def load_files():
    
    loaders = [
        DirectoryLoader(path=str(docs_path), glob="*.pdf", loader_cls=PyPDFLoader)
    ]
    docs = [document for loader in loaders for document in loader.lazy_load()]
    return docs


if __name__ == "__main__":
    documents = load_files() 
    print(f"Step 1 - Files are loaded and converted to documents. Total documents: {len(documents)}")

    

