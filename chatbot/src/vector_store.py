from dotenv import load_dotenv
import os

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from langchain_community.document_loaders import TextLoader, DirectoryLoader, PyPDFLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter


load_dotenv()
os.getenv("OPENAI_API_KEY") # Ensure the API key is loaded

def build_vector_store(documents, persist_directory="./chroma_store"):
    """
    Build and return a Chroma vector store from the provided documents.

    Args:
        documents (list): A list of documents to be added to the vector store.
        persist_directory (str): Directory where the vector store will be persisted.    

    Returns:
        Chroma: The built Chroma vector store.
    """
    embeddings = OpenAIEmbeddings()
    vector_store = Chroma.from_documents(
        documents,
        embeddings,
        persist_directory=persist_directory
    )
    return vector_store

def load_vector_store(persist_directory="./chroma_store"):
    """
    Load and return a Chroma vector store from the specified directory.

    Args:
        persist_directory (str): Directory where the vector store is persisted.

    Returns:
        Chroma: The loaded Chroma vector store.
    """
    embeddings = OpenAIEmbeddings()
    vector_store = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )
    return vector_store

def retrieve_similar_documents(vector_store, query, k=5):
    """
    Retrieve similar documents from the vector store based on the query.

    Args:
        vector_store (Chroma): The Chroma vector store to query.
        query (str): The query string to find similar documents.
        k (int): The number of similar documents to retrieve.
    Returns:
        list: A list of similar documents.
    """
    similar_docs = vector_store.similarity_search(query, k=k)
    return similar_docs

# Example usage:
# documents = [...]  # Load or create your documents here
# vector_store = build_vector_store(documents)
# loaded_vector_store = load_vector_store()
# similar_docs = retrieve_similar_documents(loaded_vector_store, "Your query here")

if __name__ == "__main__":
    # Load documents (for example, from text files)
    # loader = TextLoader("../../profile_data/summary.txt")
    # documents = loader.load()

    # Load both txt and pdf files from the directory
    txt_loader = DirectoryLoader("./profile_data", glob="**/*.txt", loader_cls=TextLoader)
    pdf_loader = DirectoryLoader("./profile_data", glob="**/*.pdf", loader_cls=PyPDFLoader)
    
    txt_documents = txt_loader.load()
    pdf_documents = pdf_loader.load()
    documents = txt_documents + pdf_documents
    
    print(f"Loaded {len(documents)} documents ({len(txt_documents)} txt, {len(pdf_documents)} pdf).")

    # Split documents into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    docs = text_splitter.split_documents(documents)
    print(f"Split into {len(docs)} documents.")

    # Build vector store
    vector_store = build_vector_store(docs)

    # Retrieve similar documents
    query = "where did you go to school?"
    similar_docs = retrieve_similar_documents(vector_store, query)
    for doc in similar_docs:
        print(doc.page_content)