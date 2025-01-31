from llama_index.core import VectorStoreIndex, Document
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from experiments import query_engine_corrected_my_method
from dotenv import load_dotenv
import os
import logging
import asyncio

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rebel_test.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def create_test_index(embed_model):
    """Create a simple index with test documents."""
    documents = [
        Document(text="Transformers have revolutionized natural language processing by introducing self-attention mechanisms."),
        Document(text="BERT is a bidirectional transformer model pre-trained on masked language modeling."),
        Document(text="GPT models are autoregressive transformers trained to predict the next token."),
        Document(text="Machine learning models require large amounts of training data to perform well."),
        Document(text="Neural networks consist of layers of interconnected artificial neurons."),
        Document(text="Deep learning architectures can automatically learn hierarchical representations."),
        Document(text="Attention mechanisms allow models to focus on relevant parts of the input sequence."),
    ]
    
    return VectorStoreIndex.from_documents(
        documents,
        embed_model=embed_model,
        show_progress=True
    )

def test_rebel_rerank():
    """Test the REBEL reranking method."""
    try:
        # Initialize once for all tests
        llm = OpenAI(
            temperature=0,
            model="gpt-4",
            api_key=os.getenv("OPENAI_API_KEY"),
            api_base=os.getenv("OPENAI_API_BASE")
        )
        
        embed_model = OpenAIEmbedding(
            model="text-embedding-3-large",
            api_key=os.getenv("OPENAI_API_KEY"),
            api_base=os.getenv("OPENAI_API_BASE")
        )
        
        index = create_test_index(embed_model)
        query_engine = query_engine_corrected_my_method(index, llm, embed_model)
        
        # Run all queries in parallel
        test_queries = [
            "How do transformer models work?",
            "What is the role of attention in deep learning?",
            "Explain the difference between BERT and GPT"
        ]
        
        async def process_single_query(query_engine, query):
            logger.info(f"\nTest Query: {query}")
            
            try:
                # Get response
                response = query_engine.query(query)
                
                # Log results
                logger.info(f"Response: {response.response[:200]}...")
                logger.info(f"Number of source nodes: {len(response.source_nodes)}")
                
                # Log source nodes and their scores
                logger.info("Source nodes after reranking:")
                for j, node in enumerate(response.source_nodes, 1):
                    score = node.score if hasattr(node, 'score') else 'N/A'
                    logger.info(f"Node {j} (Score: {score}):")
                    logger.info(f"Text preview: {node.text[:100]}...")
                
                return True
            except Exception as e:
                logger.error(f"Error processing query '{query}': {str(e)}")
                return False
        
        async def process_queries():
            tasks = [
                asyncio.create_task(process_single_query(query_engine, query))
                for query in test_queries
            ]
            return await asyncio.gather(*tasks)
            
        results = asyncio.run(process_queries())
        
        logger.info("\nREBEL rerank tests completed successfully")
        return all(results)
        
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_rebel_rerank()
    exit(0 if success else 1) 