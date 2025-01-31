from llama_index.core import VectorStoreIndex, Document
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from experiments import *
from dotenv import load_dotenv
import os
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('experiment_validation.log'),
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
    ]
    
    return VectorStoreIndex.from_documents(documents, embed_model=embed_model)

def test_experiment(name, query_engine_func, index, llm, embed_model):
    """Test a single experiment configuration."""
    logger.info(f"\nTesting experiment: {name}")
    try:
        # Initialize query engine
        query_engine = query_engine_func(index, llm, embed_model)
        
        # Test with a simple query
        query = "How do transformer models work?"
        response = query_engine.query(query)
        
        # Validate response
        if not response or not response.response:
            raise ValueError("Query returned empty response")
        
        # Log success and preview
        logger.info("✓ Query engine initialized and responded successfully")
        logger.info(f"Preview response: {response.response[:100]}...")
        logger.info(f"Number of source nodes: {len(response.source_nodes)}")
        
        return True
    except Exception as e:
        logger.error(f"✗ Experiment '{name}' failed: {str(e)}")
        return False

def validate_experiments():
    """Validate all experimental configurations."""
    try:
        # Load environment variables
        load_dotenv()
        
        # Check for required API keys
        required_keys = {
            "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
            "COHERE_API_KEY": os.getenv("COHERE_API_KEY")
        }
        
        missing_keys = [key for key, value in required_keys.items() if not value]
        if missing_keys:
            raise ValueError(f"Missing required API keys: {', '.join(missing_keys)}")
        
        # Initialize models
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
        
        # Create test index
        logger.info("Creating test index...")
        index = create_test_index(embed_model)
        
        # Define all experiments to test
        experiments = {
            # "Vanilla": query_engine_naive,
            "Cohere Rerank": query_engine_rerank,
            # "LLM Rerank": query_engine_llm_rerank,
            # "Static Rerank": query_engine_wholistic_rerank,
            "REBEL Method": query_engine_corrected_my_method,
            # "HyDE": query_engine_hyde,
            # "HyDE + LLM Rerank": query_engine_hyde_llm_rerank,
            # "HyDE + Cohere Rerank": query_engine_hyde_rerank,
            # "HyDE + Static Rerank": query_engine_hyde_wholistic_rerank,
            # "HyDE + REBEL": query_engine_hyde_corrected_my_method,
            # "MMR": query_engine_mmr,
            # "MMR + HyDE": query_engine_mmr_hyde
        }
        
        # Test each experiment
        results = {}
        for name, func in experiments.items():
            results[name] = test_experiment(name, func, index, llm, embed_model)
        
        # Summary
        total = len(experiments)
        passed = sum(results.values())
        logger.info(f"\nValidation Summary: {passed}/{total} experiments passed")
        
        if passed < total:
            logger.warning("Failed experiments:")
            for name, success in results.items():
                if not success:
                    logger.warning(f"- {name}")
        else:
            logger.info("All experiments validated successfully!")
        
        return passed == total
        
    except Exception as e:
        logger.error(f"Validation failed: {str(e)}")
        return False

if __name__ == "__main__":
    success = validate_experiments()
    exit(0 if success else 1) 