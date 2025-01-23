from build_vector_store import VectorStoreBuilder
from retrieve import get_retrieved_nodes, pretty_print, visualize_retrieved_nodes
from experiments import *
import pandas as pd
from tonic_validate import ValidateScorer, ValidateApi
from tonic_validate.metrics import RetrievalPrecisionMetric, AnswerSimilarityMetric


def remove_nul_chars_from_run_data(run_data):
    def remove_nul_chars_from_string(s):
        """Remove NUL characters from a single string."""
        return s.replace('\x00', '')
    """Iterate over all fields of RunData to remove NUL characters."""
    for run in run_data:
        run.reference_question = remove_nul_chars_from_string(run.reference_question)
        run.reference_answer = remove_nul_chars_from_string(run.reference_answer)
        run.llm_answer = remove_nul_chars_from_string(run.llm_answer)
        run.llm_context = [remove_nul_chars_from_string(context) for context in run.llm_context]


def make_get_llama_response(query_engine):
    def get_llama_response(prompt):
        # print(prompt)
        response = query_engine.query(prompt)
        context = []
        for x in response.source_nodes:
            # Initialize context string with the text of the node
            node_context = x.text
            # Check if 'window' metadata exists and append it to the context
            if 'window' in x.metadata:
                node_context += "\n\nWindow Context:\n" + x.metadata['window']
            context.append(node_context)
        return {
            "llm_answer": response.response,
            "llm_context_list": context
        }
    return get_llama_response


def run_experiment(experiment_name, query_engine, scorer, benchmark, validate_api, project_key, upload_results=False, runs=5):
    # List to store results dictionaries
    results_list = []

    for i in range(runs):
        get_llama_response_func = make_get_llama_response(query_engine)
        run = scorer.score(benchmark,
                           get_llama_response_func,
                           callback_parallelism=1,
                           scoring_parallelism=1)
        for _ in range(100):
            print(f"{experiment_name} Run {i+1} Overall Scores:", run.overall_scores)
        remove_nul_chars_from_run_data(run.run_data)

        # Add this run's results to the list
        results_list.append({'Run': i+1, 'Experiment': experiment_name, 'OverallScores': run.overall_scores})

        if upload_results:
            validate_api.upload_run(project_key, run=run, run_metadata={"approach": experiment_name, "run_number": i+1})
        else:
            print(f"Skipping upload for {experiment_name} Run {i+1}.")

    # Create a DataFrame from the list of results dictionaries
    results_df = pd.DataFrame(results_list)

    # Return the DataFrame containing all the results
    return results_df

def summarize_experiment_results(results_df):
    # Extract overall_scores fields into separate columns
    results_df['AnswerSimilarity'] = results_df['OverallScores'].apply(lambda x: x['answer_similarity'])
    results_df['RetrievalPrecision'] = results_df['OverallScores'].apply(lambda x: x['retrieval_precision'])

    # Group by Experiment and calculate the mean and standard deviation of the required metrics
    summary_df = results_df.groupby('Experiment').agg(
        MeanRetrievalPrecision=('RetrievalPrecision', 'mean'),
        StdRetrievalPrecision=('RetrievalPrecision', 'std'),
        MeanAnswerSimilarity=('AnswerSimilarity', 'mean'),
        StdAnswerSimilarity=('AnswerSimilarity', 'std')
    ).reset_index()

    # Print the summary
    print("Experiment Summary:")
    print(summary_df.to_string(index=False))

def main():
    tonic_validate_api_key = os.getenv("TONIC_VALIDATE_API_KEY")
    tonic_validate_project_key = tonic_validate_benchmark_key = os.getenv("TONIC_VALIDATE_PROJECT_KEY")

    validate_api = ValidateApi(tonic_validate_api_key)
    benchmark = validate_api.get_benchmark(tonic_validate_benchmark_key)
    scorer = ValidateScorer(metrics=[RetrievalPrecisionMetric(), AnswerSimilarityMetric()], model_id="gpt-4")

    vector_store_builder = VectorStoreBuilder(model_name='gpt-4o')
    vector_store_builder.download_dataset()
    index = vector_store_builder.create_vector_store()

    llm = vector_store_builder.llm
    embed_model = vector_store_builder.embed_model

    query_engine = index.as_query_engine(embed_model=vector_store_builder.embed_model)
    response = query_engine.query("What is the main idea of the paper about red-teaming?")
    print(response)

    experiments = { # comment out the ones you don't want to run
        "Vanilla": query_engine_naive(index, llm, embed_model),
        # 'VDB + Cohere rerank': query_engine_rerank,
        # 'VDB + LLM Rerank': query_engine_llm_rerank,
        # 'VDB + Static Rerank': query_engine_wholistic_rerank,
        # 'VDB + Corrected Our Method': query_engine_corrected_my_method,
        # 'VDB + HyDE': query_engine_hyde,
        # "VDB + HyDE + LLM Rerank": query_engine_hyde_llm_rerank,
        # "VDB + HyDE + Cohere Rerank": query_engine_hyde_rerank,
        # 'VDB + HyDE + Static Rerank': query_engine_hyde_wholistic_rerank,
        # 'VDB + HyDE + Corrected Our Method': query_engine_hyde_corrected_my_method,
        # 'VDB + MMR' : query_engine_mmr,
        # 'VDB + MMR + HyDE' : query_engine_mmr_hyde,
    }

    for experiment_name, query_engine in experiments.items():
        experiment_results_df = run_experiment(experiment_name,
                                           query_engine,
                                           scorer,
                                           benchmark,
                                           validate_api,
                                           tonic_validate_project_key,
                                           upload_results=False,
                                           runs=5)
        all_experiments_results_df = pd.concat([all_experiments_results_df, experiment_results_df], ignore_index=True)

    summarize_experiment_results(all_experiments_results_df)
    pd.to_csv("all_experiments_results_df.csv", all_experiments_results_df)


if __name__ == "__main__":
    main()