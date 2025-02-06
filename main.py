from process_documents import process_documents, check_vector_store
from experiments import *
import pandas as pd
from tonic_validate import ValidateScorer, ValidateApi
from tonic_validate.metrics import RetrievalPrecisionMetric, AnswerSimilarityMetric
import logging
import os
from datetime import datetime
import json
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding 
import chromadb
import argparse

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'experiments_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run RAG experiments with vector store')
    parser.add_argument('--force', 
                       action='store_true',
                       help='Force reprocessing of documents even if vector store exists')
    parser.add_argument('--testing',
                       action='store_true',
                       help='Run in testing mode with limited documents')
    parser.add_argument('--upload',
                       action='store_true',
                       help='Upload results to Tonic Validate')
    parser.add_argument('--runs',
                       type=int,
                       default=5,
                       help='Number of runs per experiment (default: 5)')
    return parser.parse_args()

def setup_validate():
    """Set up Tonic Validate components."""
    tonic_validate_api_key = os.getenv("TONIC_VALIDATE_API_KEY")
    tonic_validate_project_key = os.getenv("TONIC_VALIDATE_PROJECT_KEY")
    tonic_validate_benchmark_key = os.getenv("TONIC_VALIDATE_BENCHMARK_KEY")
    
    if not tonic_validate_api_key or not tonic_validate_project_key:
        raise ValueError("Missing Tonic Validate API keys in environment variables")
    
    validate_api = ValidateApi(tonic_validate_api_key)
    benchmark = validate_api.get_benchmark(tonic_validate_benchmark_key)
    
    # Configure the scorer to use the same API endpoint as our LLM
    scorer = ValidateScorer(
        metrics=[RetrievalPrecisionMetric(), AnswerSimilarityMetric()],
        model_id="gpt-4",
    )
    
    return validate_api, benchmark, scorer, tonic_validate_project_key

def make_get_llama_response(query_engine):
    """Create response function for the query engine."""
    def get_llama_response(prompt):
        try:
            response = query_engine.query(prompt)
            context = []
            for x in response.source_nodes:
                node_context = x.text
                if 'window' in x.metadata:
                    node_context += f"\n\nWindow Context:\n{x.metadata['window']}"
                context.append(node_context)
            return {
                "llm_answer": response.response,
                "llm_context_list": context
            }
        except Exception as e:
            logger.error(f"Error in get_llama_response: {str(e)}")
            return {
                "llm_answer": "Error processing query",
                "llm_context_list": []
            }
    return get_llama_response

def run_experiment(experiment_name, query_engine, scorer, benchmark, validate_api, project_key, upload_results=False, runs=5):
    """Run a single experiment with multiple runs."""
    logger.info(f"Starting experiment: {experiment_name}")
    results_list = []
    output_path = 'experiment_results'
    os.makedirs(output_path, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Load existing results if any
    results_file = f'{output_path}/full_results_{timestamp}.csv'
    if os.path.exists(results_file):
        existing_results = pd.read_csv(results_file)
        results_list = existing_results.to_dict('records')
    
    for i in range(runs):
        try:
            logger.info(f"Starting {experiment_name} - Run {i+1}/{runs}")
            get_llama_response_func = make_get_llama_response(query_engine)
            
            run = scorer.score(
                benchmark,
                get_llama_response_func,
                callback_parallelism=1,
                scoring_parallelism=1
            )
            
            # Log scores
            logger.info(f"{experiment_name} Run {i+1} Scores: {run.overall_scores}")
            
            # Clean data
            if hasattr(run, 'run_data'):
                for data in run.run_data:
                    for field in ['reference_question', 'reference_answer', 'llm_answer']:
                        if hasattr(data, field):
                            setattr(data, field, getattr(data, field).replace('\x00', ''))
                    if hasattr(data, 'llm_context'):
                        data.llm_context = [ctx.replace('\x00', '') for ctx in data.llm_context]
            
            # Store results
            run_result = {
                'Run': i+1,
                'Experiment': experiment_name,
                'OverallScores': run.overall_scores
            }
            results_list.append(run_result)
            
            # Save results after each run
            current_results = pd.DataFrame(results_list)
            current_results.to_csv(results_file, index=False)
            summarize_results(current_results, output_path)
            
            # Upload if requested
            if upload_results:
                validate_api.upload_run(
                    project_key,
                    run=run,
                    run_metadata={"approach": experiment_name, "run_number": i+1}
                )
                logger.info(f"Uploaded results for {experiment_name} Run {i+1}")
            
        except Exception as e:
            logger.error(f"Error in {experiment_name} Run {i+1}: {str(e)}")
            run_result = {
                'Run': i+1,
                'Experiment': experiment_name,
                'OverallScores': {'error': str(e)}
            }
            results_list.append(run_result)
            # Save results even after error
            current_results = pd.DataFrame(results_list)
            current_results.to_csv(results_file, index=False)
            summarize_results(current_results, output_path)
    return


def summarize_results(results_df, output_path='experiment_results'):
    """Summarize and save experiment results."""
    os.makedirs(output_path, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Extract metrics
    results_df['AnswerSimilarity'] = results_df['OverallScores'].apply(
        lambda x: x.get('answer_similarity', None) if isinstance(x, dict) else None)
    results_df['RetrievalPrecision'] = results_df['OverallScores'].apply(
        lambda x: x.get('retrieval_precision', None) if isinstance(x, dict) else None)
    
    # Calculate summary statistics
    summary = results_df.groupby('Experiment').agg({
        'RetrievalPrecision': ['mean', 'std'],
        'AnswerSimilarity': ['mean', 'std']
    }).round(4)
    
    # Save summary
    summary.to_csv(f'{output_path}/summary_{timestamp}.csv')
    
    # Log summary
    logger.info("\nExperiment Summary:")
    logger.info("\n" + str(summary))
    
    return summary

def main():
    """Main function to run all experiments."""
    args = parse_args()
    
    try:
        # Check environment variables
        api_key = os.getenv("OPENAI_API_KEY")
        api_base = os.getenv("OPENAI_API_BASE")
        
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        if not api_base:
            raise ValueError("OPENAI_API_BASE environment variable is not set")
            
        logger.info("Initializing with API base: %s", api_base)
        
        # Initialize LLM and embedding model with proper error handling
        try:
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
            logger.info("Successfully initialized OpenAI clients with custom API base")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI clients: {str(e)}")
            raise

        # Check if vector store needs to be initialized
        if check_vector_store() and not args.force:
            logger.info("Loading existing vector store...")
            index = process_documents(force_reprocess=False, testing=args.testing)
        else:
            if args.force:
                logger.info("Force flag set. Initializing new vector store...")
            else:
                logger.info("No existing vector store found. Initializing new vector store...")
            index = process_documents(force_reprocess=True, testing=args.testing)
        
        # Setup validation components
        validate_api, benchmark, scorer, project_key = setup_validate()
        
        # Define experiments
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
        
        # Run experiments
        for name, engine in experiments.items():
            run_experiment(
                name, engine, scorer, benchmark, validate_api, project_key,
                upload_results=args.upload, runs=args.runs
            )
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()