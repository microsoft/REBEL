# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from llama_index.core import VectorStoreIndex, Document
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from experiments import *
from dotenv import load_dotenv
import os
import logging
import time
import pandas as pd
from process_documents import process_documents, check_vector_store

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

def measure_inference_time(query_engine, query):
    """Measure inference time for a single query with detailed breakdowns."""
    timings = {}
    
    # Measure total query time
    start_time = time.time()
    response = query_engine.query(query)
    total_time = time.time() - start_time
    timings['total'] = total_time
    
    return timings, response

def test_single_step():
    """Test each method for a single step and measure inference time."""
    try:
        # Load environment variables
        load_dotenv()
        
        # Check environment variables
        api_key = os.getenv("OPENAI_API_KEY")
        api_base = os.getenv("OPENAI_API_BASE")
        
        if not api_key or not api_base:
            raise ValueError("OPENAI_API_KEY and OPENAI_API_BASE must be set")
        
        logger.info("Initializing with API base: %s", api_base)
        
        # Initialize models
        llm = OpenAI(
            temperature=0,
            model="gpt-4o",
            api_key=api_key,
            api_base=api_base
        )
        
        embed_model = OpenAIEmbedding(
            model="text-embedding-3-large",
            api_key=api_key,
            api_base=api_base
        )
        
        # Load or create vector store
        if check_vector_store():
            logger.info("Loading existing vector store...")
            index = process_documents(force_reprocess=False, testing=False)  # Use full index
        else:
            logger.info("Initializing new vector store...")
            index = process_documents(force_reprocess=True, testing=False)  # Use full index
        
        # Define experiments to test
        experiments = {
            "No Rerank": query_engine_no_rerank(index, llm, embed_model),
            "Cohere Rerank": query_engine_cohere_rerank(index, embed_model),
            "LLM Rerank": query_engine_llm_rerank(index, llm, embed_model),
            "One-Turn REBEL Rerank": query_engine_one_turn_rebel_rerank(index, llm, embed_model),
            "Two-Turn Relevance-Only REBEL Rerank": query_engine_two_turn_relevance_only_rebel_rerank(index, llm, embed_model),
            "Two-Turn REBEL Rerank": query_engine_two_turn_rebel_rerank(index, llm, embed_model),
            "HyDE": query_engine_hyde(index, llm, embed_model),
            "MMR": query_engine_mmr(index, llm, embed_model),
        }
        
        # Test query
        test_query = "What are the key components of a transformer model architecture?"
        
        # Measure inference times
        results = []
        for name, engine in experiments.items():
            try:
                logger.info(f"\nTesting {name}...")
                
                # Measure inference time
                timings, response = measure_inference_time(engine, test_query)
                
                results.append({
                    'Method': name,
                    'Total Time (s)': timings['total'],
                    'Response Length': len(response.response),
                    'Num Sources': len(response.source_nodes),
                    'Tokens Per Second': len(response.response) / timings['total']
                })
                
                # Log detailed results
                logger.info(f"✓ Success:")
                logger.info(f"  Total Time: {timings['total']:.2f}s")
                logger.info(f"Response length: {len(response.response)} chars")
                logger.info(f"Number of sources: {len(response.source_nodes)}")
                logger.info(f"Characters Per Second: {len(response.response) / timings['total']:.2f}")
                logger.info(f"Preview response: {response.response[:200]}...")
                logger.info("-" * 80)
                
            except Exception as e:
                logger.error(f"✗ Error testing {name}: {str(e)}")
                results.append({
                    'Method': name,
                    'Total Time (s)': None,
                    'Response Length': None,
                    'Num Sources': None,
                    'Tokens Per Second': None
                })
        
        # Create DataFrame and save results
        df = pd.DataFrame(results)
        df = df.sort_values('Total Time (s)')
        
        # Save results to CSV
        df.to_csv('inference_times.csv', index=False)
        logger.info("\nResults saved to inference_times.csv")
        
        # Print summary
        print("\nInference Time Summary:")
        print(df.to_string(index=False))
        
        return df
        
    except Exception as e:
        logger.error(f"Error in test_single_step: {str(e)}")
        raise

if __name__ == "__main__":
    test_single_step() 
