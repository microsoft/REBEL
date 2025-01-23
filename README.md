# REBEL: RErank BEyond reLevance

REBEL is a novel framework that enhances Retrieval-Augmented Generation (RAG) systems by incorporating query-dependent, multi-criteria document reranking. While traditional RAG systems focus solely on relevance when selecting documents, REBEL recognizes that other factors like recency, credibility, and perspective diversity can be crucial for generating high-quality responses.

## How REBEL Works

REBEL uses a two-step approach:

1. **Meta-Prompt Generation**: For each user query, REBEL first analyzes the query to infer what secondary criteria (beyond relevance) would be important. For example:
   - A query about current medical treatments might prioritize recency and authoritativeness
   - A query about economic policies might prioritize perspective diversity and reasoning depth
   - A technical query might prioritize specificity and accuracy

2. **Multi-Criteria Reranking**: Using these inferred criteria, REBEL generates a custom reranking prompt that:
   - Scores documents on both relevance (0-10) and secondary criteria (0-5)
   - Computes a weighted composite score combining all factors
   - Ranks documents based on this comprehensive assessment

This approach enables REBEL to select context documents that are not just relevant, but optimized for generating high-quality responses to each specific query type.

## Key Features

- **Query-Dependent Criteria**: Unlike static approaches, REBEL dynamically infers what properties matter most for each query
- **Chain-of-Thought Integration**: Uses a meta-prompting strategy similar to Chain-of-Thought techniques
- **Plug-and-Play**: Requires no model fine-tuning or architectural changes to existing RAG pipelines
- **Interpretable**: Provides clear scoring rubrics and weightings for all criteria

## Experimental Results

Our experiments show that REBEL:
- Significantly improves answer quality compared to relevance-only approaches
- Maintains high retrieval precision while optimizing for multiple criteria
- Outperforms both standard RAG and static multi-criteria reranking methods

## This Repository

This repository contains code for reproducing the experimental results presented in our paper. These experiments can be run via:
```
export OPENAI_API_KEY=<your-api-key>
export OPENAI_API_BASE=<your-api-base> (optional)
python main.py
```

## Citation

If you use REBEL in your research, please cite our paper:
```bibtex
@article{levine2024relevance,
  title={Relevance Isn't All You Need: Leveraging Chain-of-Thought To Generate Query-Dependent Multi-Criteria Reranking Prompts For Retrieval},
  author={LeVine, Will and Varjavand, Bijan},
  year={2024}
}
```

## Contact

For questions or feedback about REBEL, please contact Will LeVine (levinewill@icloud.com).
