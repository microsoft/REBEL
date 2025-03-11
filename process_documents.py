# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from datasets import load_dataset
import chromadb
import os
from tqdm import tqdm
import logging
import shutil

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('document_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def check_vector_store(db_path='./data/chroma_db'):
    """Check if vector store exists and has documents."""
    if not os.path.exists(db_path):
        logger.info("No existing vector store found")
        return False
        
    try:
        # Initialize ChromaDB client
        client = chromadb.Client(
            chromadb.config.Settings(
                is_persistent=True,
                persist_directory=db_path,
                anonymized_telemetry=False
            )
        )
        
        # Get or create collection
        collection = client.get_or_create_collection(
            name="ai_arxiv_papers",
            metadata={"hnsw:space": "cosine"}
        )
        
        count = collection.count()
        logger.info(f"Found existing vector store with {count} documents")
        return count > 0
    except Exception as e:
        logger.warning(f"Error checking vector store: {str(e)}")
        return False

def download_dataset(text_directory='./data/text', testing=False):
    """Download arxiv papers dataset if not already present."""
    if not os.path.exists(text_directory):
        logger.info("Downloading dataset...")
        os.makedirs(text_directory, exist_ok=True)
        
        dataset = load_dataset("jamescalam/ai-arxiv", split="train")
        for i, item in enumerate(tqdm(dataset)):
            filename = f"{text_directory}/{item['title'].replace('/', '_')}.txt"
            with open(filename, "w") as fp:
                fp.write(item["content"])
            
            if testing and i >= 10:
                break
    else:
        logger.info("Dataset already downloaded")

def process_documents(text_directory='./data/text', db_path='./data/chroma_db', testing=False, force_reprocess=False):
    """Process documents and create vector store."""
    # Ensure OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY environment variable is not set")

    # If force_reprocess, delete existing database
    if force_reprocess and os.path.exists(db_path):
        logger.info(f"Force reprocess requested. Removing existing database at {db_path}")
        shutil.rmtree(db_path)
        logger.info("Existing database removed")

    # Initialize embedding model
    embed_model = OpenAIEmbedding(
        model="text-embedding-3-large",
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_API_BASE", None)
    )

    # Set up ChromaDB with explicit settings
    client = chromadb.Client(
        chromadb.config.Settings(
            is_persistent=True,
            persist_directory=db_path,
            anonymized_telemetry=False
        )
    )
    
    # Get or create collection
    collection = client.get_or_create_collection(
        name="ai_arxiv_papers",
        metadata={"hnsw:space": "cosine"}
    )
    
    # If collection already has documents and we're not forcing reprocess, load existing index
    if collection.count() > 0 and not force_reprocess:
        logger.info("Loading existing vector store...")
        vector_store = ChromaVectorStore(chroma_collection=collection)
        return VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)

    # Process documents
    logger.info("Creating new vector store...")
    
    # Download dataset if needed
    download_dataset(text_directory, testing)
    
    # Load documents
    logger.info("Loading documents...")
    documents = SimpleDirectoryReader(text_directory).load_data()
    logger.info(f"Loaded {len(documents)} documents")

    # Create nodes with smaller chunk size for better processing
    logger.info("Processing documents into nodes...")
    node_parser = SentenceSplitter(
        chunk_size=2000,
        chunk_overlap=200
    )
    nodes = node_parser.get_nodes_from_documents(documents)
    logger.info(f"Created {len(nodes)} nodes")

    # Create vector store
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Build index
    logger.info("Building vector store index...")
    index = VectorStoreIndex(
        nodes,
        storage_context=storage_context,
        embed_model=embed_model,
        show_progress=True
    )
    
    logger.info("Vector store creation complete")
    return index

def main(testing=False):
    """Main function to run the document processing pipeline."""
    try:
        # Download dataset if needed
        download_dataset(testing=testing)
        
        # Process documents and create vector store
        index = process_documents(testing=testing)
        
        # Test the index with a simple query
        query_engine = index.as_query_engine()
        response = query_engine.query(
            "What are some key papers about large language models?"
        )
        logger.info(f"Test query response: {response}")
        
    except Exception as e:
        logger.error(f"Error in processing: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main(testing=False) 
