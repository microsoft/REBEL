# REBEL: RErank BEyond reLevance 
<a target="_blank" href="https://colab.research.google.com/github/levinwil/REBEL/blob/main/REBEL.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

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

## Setup

1. Clone the repository:
```bash
git clone https://github.com/levinwil/REBEL.git
cd REBEL
```

2. Install dependencies:
```bash
pip install -e git+https://github.com/bvarjavand/tonic_validate.git
git clone https://github.com/bvarjavand/llama_index.git
cd llama_index/llama-index-core && pip install -e .
```

3. Create a `.env` file with your API keys:
```bash
OPENAI_API_KEY=your-api-key-here
OPENAI_API_BASE=https://your-api-base-url-here  # Optional: Only if using a proxy
LITELLM_API_KEY=your-litellm-key-here
LITELLM_BASE_URL=https://your-litellm-base-url-here  # Optional: Only if using a proxy. same as OPENAI_API_BASE
COHERE_API_KEY=your-cohere-key-here  # Required for Cohere rerank experiments
TONIC_VALIDATE_API_KEY=your-tonic-key-here
TONIC_VALIDATE_PROJECT_KEY=your-project-key-here
TONIC_VALIDATE_BENCHMARK_KEY=your-benchmark-key-here
```

## Usage

The main script supports several command-line arguments:

```bash
# Run with default settings
python main.py

# Force reprocessing of documents
python main.py --force

# Run in testing mode (processes fewer documents)
python main.py --testing

# Upload results to Tonic Validate
python main.py --upload

# Specify number of runs per experiment
python main.py --runs 3
```

To get inference time data, run the `test_experiments.py` file. This is also useful for testing these methods with different vector databases.

```bash
# Run normally
python test_experiments.py
```

## Experiments

The framework includes several RAG strategies:

1. Vanilla RAG
2. RAG with Cohere reranking
3. RAG with LLM reranking
4. RAG with static reranking
5. RAG with REBEL method
6. HyDE (Hypothetical Document Embeddings)

## Project Structure

- `main.py`: Main script for running experiments
- `process_documents.py`: Handles document processing and vector store creation
- `experiments.py`: Contains different RAG implementations
- `test_experiments.py`: Runs a single run for each experiment and saves inference times

## Vector Store

The project uses ChromaDB as the vector store. The store is:
- Created automatically on first run
- Persisted to disk for reuse
- Can be forced to rebuild using `--force`
- Configurable chunk size and overlap

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## License

MIT License
Copyright (c) 2024 REBEL Contributors
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
