import streamlit as st
import os
import chromadb
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.ollama import Ollama
import time

# Set Paths
PDF_FOLDER = r"D:\Demo_Prep"  # Data storage folder
CHROMA_DB_FOLDER = r"D:\Demo_Prep\ChromaDB"  # ChromaDB storage folder

# Ensure ChromaDB folder exists
os.makedirs(CHROMA_DB_FOLDER, exist_ok=True)

# Dictionary of PDFs
pdf_dic = {
    "summary.pdf": "summary_index", 
    "Family.pdf": "family_index",
    "TI_S.pdf": "tis_index",
    "Year Old Records.pdf": "year_old_index",
    "FAQS.pdf": "faqs_index",
    "links.pdf": "links_index"
}

# Initialize ChromaDB client
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_FOLDER)

# Function to Load and Process Each PDF Separately
@st.cache_resource
def load_and_embed_pdf(pdf_filename, index_name):
    """Loads a single PDF, embeds it, and creates a ChromaDB index."""
    pdf_path = os.path.join(PDF_FOLDER, pdf_filename)
    documents = SimpleDirectoryReader(input_files=[pdf_path]).load_data()
    
    # Load Embedding Model
    embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Load or Create ChromaDB Collection
    vector_store = ChromaVectorStore(chroma_client=chroma_client, collection_name=index_name)
    index = VectorStoreIndex.from_documents(documents, vector_store=vector_store, embed_model=embed_model)
    
    return index

# Load each PDF separately into its own ChromaDB index
indices = {pdf: load_and_embed_pdf(pdf, index_name) for pdf, index_name in pdf_dic.items()}

# Initialize Llama 3.2 Local Model using Ollama
llm = Ollama(model="llama3.2", temperature=0.2)

# PDF Page Mapping
pdf_page_name = {
    0: "summary.pdf", 
    2: "Family.pdf",
    1: "TI_S.pdf",
    3: "Year Old Records.pdf",
    4: "FAQS.pdf",
    5: "links.pdf"
}

# Streamlit UI
st.title("CDP GPT")
query = st.text_input("Enter your query:")

if query:
    start_time = time.time()
    
    index = indices['summary.pdf']  # Default index (modify if needed)
    retriever = index.as_retriever(similarity_top_k=1)
    retrieved_docs = retriever.retrieve(query)
    
    for doc in retrieved_docs:
        page_number = int(doc.metadata.get('page_label', '0'))
        pdf_name = pdf_page_name.get(page_number, "summary.pdf")
        
        if "link" in query.lower():
            pdf_name = "links.pdf"
        
        index = indices[pdf_name]
        retriever = index.as_retriever(similarity_top_k=3)
        retrieved_docs = retriever.retrieve(query)
        
        extracted_text = " ".join([doc.node.text for doc in retrieved_docs])
        
        response = llm.complete(f"Search within the following text and provide an answer. Here is the text: {extracted_text}\n\nQuery: {query}. "
                                "Strictly refer to the above context before answering. If unclear, ask for clarification.")
        
        st.write(response.text)
        st.write(f"Response Time: {time.time() - start_time:.2f} sec")
