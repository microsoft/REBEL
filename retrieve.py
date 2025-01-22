from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core import QueryBundle
from llama_index.core.postprocessor import REBELRerank

import pandas as pd
import torch
from IPython.display import display, HTML


def get_retrieved_nodes(
    query_str,
    index,
    llm,
    vector_top_k=10,
    reranker_top_n=3,
    with_reranker=False,
):
    query_bundle = QueryBundle(query_str)
    # configure retriever
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=vector_top_k,
    )
    retrieved_nodes = retriever.retrieve(query_bundle)
    retrieved_nodes.reverse()

    if with_reranker:
        reranker = REBELRerank(
            llm=llm, top_n=reranker_top_n
        )
        retrieved_nodes = reranker.postprocess_nodes(
            retrieved_nodes, query_bundle
        )

        # clear cache, rank_zephyr uses 16GB of GPU VRAM
        del reranker
        torch.cuda.empty_cache()

    return retrieved_nodes


def pretty_print(df):
    return display(HTML(df.to_html().replace("\\n", "<br>")))


def visualize_retrieved_nodes(nodes) -> None:
    result_dicts = []
    for node in nodes:
        result_dict = {"Score": node.score, "Text": node.node.get_text()}
        result_dicts.append(result_dict)

    pretty_print(pd.DataFrame(result_dicts))