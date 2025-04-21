import streamlit as st
import os
import faiss
from llama_index.core import VectorStoreIndex, Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.llms.ollama import Ollama
from llama_index.core.llms import ChatMessage, MessageRole
import httpx
import re
import time
import json
from llama_index.core import SimpleDirectoryReader

keywords = [
    "website", "webpage", "site", "URL", "link", "web link", "web reference", 
    "hyperlink", "domain", "homepage", "official website", "source", "reference", 
    "resource", "redirect", "click here", "browse", "navigate", "open", 
    "web address", "site address", "permalink", "portal", "platform", "blog", "forum"
    ]

client = httpx.Client(timeout=httpx.Timeout(300000000000000000000.0)) 

# Set Paths
DATA_FOLDER = r"D:\text_masti"   # Data is stored 
FAISS_INDEX_FOLDER = r"D:\text_masti\FAISS_Indices"   # Index to store

# Ensure FAISS index folder exists
os.makedirs(FAISS_INDEX_FOLDER, exist_ok=True)

# Dictionary of Data Files
data_dic = {
    "summary": "faiss_summary.bin",
    "Family": "faiss_family.bin",
    "TI_S": "faiss_tis.bin",
    "Year_Old_Records": "faiss_year_old.bin",
    "FAQS": "faiss_faqs.bin",
    "links": "faiss_links.bin"
}

# Function to Load and Process Each Data File Separately
@st.cache_resource
def load_and_embed_data(filename, index_filename):
    """Loads a single data file (PDF, JSON, or text), embeds it, and creates a FAISS index."""
    file_path = os.path.join(DATA_FOLDER, filename)
    index_path = os.path.join(FAISS_INDEX_FOLDER, index_filename)
    
    # Load Data
    if filename.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8") as file:
            data = file.read()
    elif filename.endswith(".json"):
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.dumps(json.load(file))
    elif filename.endswith(".pdf"):
        documents = SimpleDirectoryReader(input_files=[file_path]).load_data()
        data = " ".join([doc.text for doc in documents])
    else:
        raise ValueError("Unsupported file format")
    
    # Load Embedding Model
    embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Load or Create FAISS Index
    embedding_dim = 384
    faiss_index = faiss.IndexFlatL2(embedding_dim)

    if os.path.exists(index_path):
        faiss_index = faiss.read_index(index_path)

    vector_store = FaissVectorStore(faiss_index)
    index = VectorStoreIndex.from_documents([Document(text=data)], vector_store=vector_store, embed_model=embed_model)


    # Save FAISS Index
    faiss.write_index(faiss_index, index_path)
    return index

# Load each data file separately into its own index
indices = {file: load_and_embed_data(file + ext, index_file) for file, index_file in data_dic.items() for ext in [".txt", ".json", ".pdf"] if os.path.exists(os.path.join(DATA_FOLDER, file + ext))}

# Initialize Llama 3:2 Local Model using Ollama
llm = Ollama(model="llama3.2", temperature=0.2)

# Data File Page Mapping
data_page_name = {
    0: "summary",
    2: "Family",
    1: "TI_S",
    3: "Year_Old_Records",
    4: "FAQS",
    5: "links"
}

# Streamlit UI
st.title("CDP GPT")

querry_list = []
query = st.text_input("Enter your query:")
if query:
    querry_list.append(query)
    s = time.time()
    index = indices['summary']  # Default index (modify if needed)
    query_engine = index.as_query_engine(llm=llm)
    retriever = index.as_retriever(similarity_top_k=1)
    retrieved_docs = retriever.retrieve(query)

    for doc in retrieved_docs:
        page_number = int(doc.metadata.get('page_label', '0'))
        data_name = data_page_name.get(page_number)  # Retrieving particular data file that contains relevant data
        print(data_name)
        if any(word in query.lower() for word in keywords):
            data_name = "links"
        if data_name == "links":
            query = query + " Provide Link"
        index = indices[data_name]  # Index of that data file
        retriever = index.as_retriever(similarity_top_k=3)
        retrieved_docs = retriever.retrieve(query)
        extracted_text = " ".join([doc.node.text for doc in retrieved_docs])
        e_r = time.time()  # End time to fetch required information
        
        response = llm.complete(f"Search within the following text and provide an answer. Here is the text: {extracted_text}\n\nQuery: {query} and remember this "
                                "YOU MUST strictly refer to the above context before answering. If the answer is unclear, request clarification instead of guessing. "
                                "If the retrieved context lacks sufficient details, clearly state that rather than making assumptions.")
        e = time.time()
        
        st.write(response.text)
        st.write(e - s)