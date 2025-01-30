from llama_index.core import VectorStoreIndex
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
import os
import logging
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('vector_store_test.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_vector_store(db_path='./data/chroma_db'):
    """Load the vector store and return both the collection and index."""
    try:
        # Initialize embedding model
        embed_model = OpenAIEmbedding(
            model="text-embedding-3-large",
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_API_BASE", None)
        )

        # Load ChromaDB
        db = chromadb.PersistentClient(path=db_path)
        collection = db.get_collection("ai_arxiv_papers")
        
        # Create vector store and index
        vector_store = ChromaVectorStore(chroma_collection=collection)
        index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)
        
        return collection, index
    except Exception as e:
        logger.error(f"Error loading vector store: {str(e)}")
        raise

def run_basic_tests(collection, index):
    """Run basic tests on the vector store."""
    tests_passed = 0
    total_tests = 4
    
    logger.info("Starting vector store tests...")

    # Test 1: Check if collection exists and has documents
    logger.info("Test 1: Checking collection size...")
    collection_count = collection.count()
    if collection_count > 0:
        logger.info(f"✓ Collection contains {collection_count} entries")
        tests_passed += 1
    else:
        logger.error("✗ Collection is empty")

    # Test 2: Verify basic query functionality
    logger.info("Test 2: Testing basic query...")
    try:
        query_engine = index.as_query_engine()
        response = query_engine.query("What is a large language model?")
        if response and response.response:
            logger.info("✓ Query engine returned a response")
            logger.info(f"Sample response: {response.response[:200]}...")
            tests_passed += 1
        else:
            logger.error("✗ Query returned empty response")
    except Exception as e:
        logger.error(f"✗ Query test failed: {str(e)}")

    # Test 3: Check source nodes retrieval
    logger.info("Test 3: Testing source node retrieval...")
    try:
        response = query_engine.query("What is deep learning?")
        if response.source_nodes and len(response.source_nodes) > 0:
            logger.info(f"✓ Query returned {len(response.source_nodes)} source nodes")
            tests_passed += 1
        else:
            logger.error("✗ No source nodes returned")
    except Exception as e:
        logger.error(f"✗ Source node test failed: {str(e)}")

    # Test 4: Test different query types
    logger.info("Test 4: Testing different query types...")
    test_queries = [
        "What are the main challenges in reinforcement learning?",
        "Explain the transformer architecture",
        "What is the difference between supervised and unsupervised learning?",
        "Describe recent advances in computer vision"
    ]
    
    try:
        all_queries_successful = True
        for query in tqdm(test_queries, desc="Running test queries"):
            response = query_engine.query(query)
            if not response or not response.response:
                all_queries_successful = False
                logger.error(f"✗ Query failed: {query}")
                break
        
        if all_queries_successful:
            logger.info("✓ All test queries completed successfully")
            tests_passed += 1
        else:
            logger.error("✗ Some test queries failed")
    except Exception as e:
        logger.error(f"✗ Query variation test failed: {str(e)}")

    # Summary
    logger.info(f"\nTest Summary: {tests_passed}/{total_tests} tests passed")
    return tests_passed == total_tests

def main():
    """Main function to run the tests."""
    try:
        logger.info("Loading vector store...")
        collection, index = load_vector_store()
        
        success = run_basic_tests(collection, index)
        
        if success:
            logger.info("All tests passed successfully!")
            return 0
        else:
            logger.error("Some tests failed. Check the logs for details.")
            return 1
            
    except Exception as e:
        logger.error(f"Testing failed with error: {str(e)}", exc_info=True)
        return 1

if __name__ == "__main__":
    exit(main()) 