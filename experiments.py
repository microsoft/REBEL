from llama_index.core import PromptTemplate
from llama_index.postprocessor.cohere_rerank import CohereRerank
from llama_index.core.postprocessor import LLMRerank, REBELRerank
from llama_index.core.indices.query.query_transform import HyDEQueryTransform
from llama_index.core.query_engine import TransformQueryEngine, RetrieverQueryEngine
import os
import logging

logger = logging.getLogger(__name__)

TEXT_QA_TEMPLATE = PromptTemplate("""You are an expert Q&A system that is trusted around the world for your factual accuracy.
Always answer the query using the provided context information, and not prior knowledge. Ensure your answers are fact-based and accurately reflect the context provided.
Some rules to follow:
1. Never directly reference the given context in your answer.
2. Avoid statements like 'Based on the context, ...' or 'The context information ...' or anything along those lines.
3. Focus on succinct answers that provide only the facts necessary, do not be verbose.Your answers should be max two sentences, up to 250 characters.
---------------------
{context_str}
---------------------
Given the context information and not prior knowledge, answer the query.
Query: {query_str}
Answer: """)


MULTICRITERIA_STATIC_RERANKING_PROMPT = PromptTemplate("""You are a re-ranking system. Your task is to analyze a user's query and a set of candidate documents, assign scores based on specified properties, and output the final ranking of documents.

**Inferred Properties**

1. **Depth of Content (0-5):**
- Higher scores indicate thorough detail and comprehensive coverage of the topic.
- A "5" is exceptionally in-depth with multiple facets addressed; a "0" is very superficial.

2. **Diversity of Perspectives (0-5):**
- Higher scores indicate that multiple viewpoints or angles are represented.
- A "5" means it engages with a variety of perspectives or sources; a "0" means it is entirely one-sided.

3. **Clarity and Specificity (0-5):**
- Higher scores indicate that the document presents information clearly and addresses the query with precise, unambiguous detail.
- A "5" means it is highly specific and clear, while a "0" means it is vague or overly general.

4. **Authoritativeness (0-5):**
- Higher scores indicate reputable sources, expert authorship, or recognized credibility.
- A "5" might be an extensively cited academic work or an official standard; a "0" would be an unknown or dubious source.

5. **Recency (0-5):**
- Higher scores indicate that the document references recent studies, data, or developments.
- A "5" means it is very current and up-to-date; a "0" means it is outdated or does not reference any time-sensitive information.

**Scoring Rubric**

- **Relevance (0-10):**
- A "10" means the document directly addresses the user's query, covering the key subject comprehensively.
- A "0" means it is completely off-topic.

- **Depth of Content (0-5):** Based on how detailed or thorough the document is.
- **Diversity of Perspectives (0-5):** Based on how many viewpoints or angles are presented.
- **Clarity and Specificity (0-5):** Based on how clear and precise the document is.
- **Authoritativeness (0-5):** Based on source credibility or recognized expertise.
- **Recency (0-5):** Based on how up-to-date the document is.

**Weighted Composite Score**
Final Score = Relevance + 0.5*(Depth of Content) + 0.5*(Diversity of Perspectives) + 0.5*(Clarity and Specificity) + 0.5*(Authoritativeness) + 0.5*(Recency)

**Instructions**
1. Assign Relevance to each document on a scale of 0-10. Discard any document with Relevance < 3.
2. For the remaining documents, assign scores for:
- Depth of Content (0-5)
- Diversity of Perspectives (0-5)
- Clarity and Specificity (0-5)
- Authoritativeness (0-5)
- Recency (0-5)
3. Compute each document's Final Score using the formula above.
4. Sort the documents by their Final Score in descending order.
5. If two documents end up with the same Final Score, prefer the one with higher Authoritativeness (or apply another consistent tie-breaking rule).
6. If no documents remain after filtering for Relevance, output nothing.
7. Output only the list of theselected documents with their Relevance scores, in this format (no extra text or commentary), where [score] is actually the Final Score and NOT the relevance score.
```
Doc: [document number], Relevance: [score]
```

**Example format:**
```
Document 1:
<summary of document 1>

Document 2:
<summary of document 2>

...

Document 10:
<summary of document 10>

Question: <question>
Answer:
Doc: 9, Relevance: 7
Doc: 3, Relevance: 4
Doc: 7, Relevance: 3

Let's try this now:

{context_str}
Question: {query_str}
Answer:
```""")


def query_engine_naive(index, llm, embed_model):
    return index.as_query_engine(llm=llm,
          text_qa_template=TEXT_QA_TEMPLATE,
          similarity_top_k=3,
          embed_model=embed_model)

def query_engine_rerank(index, llm, embed_model):
    cohere_rerank = CohereRerank(
        api_key=os.getenv("COHERE_API_KEY"),
        top_n=3)
    return index.as_query_engine(
        similarity_top_k=10,
        node_postprocessors=[cohere_rerank],
        text_qa_template=TEXT_QA_TEMPLATE,
        embed_model=embed_model)

def query_engine_llm_rerank(index, llm, embed_model):
    llm_rerank = LLMRerank(
        llm=llm,
        choice_batch_size=10,
        top_n=3)
    return index.as_query_engine(llm=llm,
          text_qa_template=TEXT_QA_TEMPLATE,
          similarity_top_k=10,
          node_postprocessors=[llm_rerank],
          embed_model=embed_model)

def query_engine_wholistic_rerank(index, llm, embed_model):
    llm_rerank = LLMRerank(
        choice_select_prompt=MULTICRITERIA_STATIC_RERANKING_PROMPT,
        llm=llm,
        choice_batch_size=10,
        top_n=3)
    return index.as_query_engine(llm=llm,
          text_qa_template=TEXT_QA_TEMPLATE,
          similarity_top_k=10,
          node_postprocessors=[llm_rerank],
          embed_model=embed_model)

def query_engine_corrected_my_method(index, llm, embed_model):
    rebel_rerank = REBELRerank(
        llm=llm,
        choice_batch_size=10,
        top_n=3)
    return index.as_query_engine(
        similarity_top_k=10,
        node_postprocessors=[rebel_rerank],
        text_qa_template=TEXT_QA_TEMPLATE,
        llm=llm,
        embed_model=embed_model)

def query_engine_hyde(index, llm, embed_model):
    base_query_engine = query_engine_naive(index, llm, embed_model)
    return TransformQueryEngine(base_query_engine, HyDEQueryTransform(llm=llm, include_original=True))

def query_engine_hyde_rerank(index, llm, embed_model):
    base_query_engine = query_engine_rerank(index, llm, embed_model)
    return TransformQueryEngine(base_query_engine, HyDEQueryTransform(llm=llm, include_original=True))

def query_engine_hyde_llm_rerank(index, llm, embed_model):
    base_query_engine = query_engine_llm_rerank(index, llm, embed_model)
    return TransformQueryEngine(base_query_engine, HyDEQueryTransform(llm=llm, include_original=True))

def query_engine_hyde_wholistic_rerank(index, llm, embed_model):
    base_query_engine = query_engine_wholistic_rerank(index, llm, embed_model)
    return TransformQueryEngine(base_query_engine, HyDEQueryTransform(llm=llm, include_original=True))

def query_engine_hyde_corrected_my_method(index, llm, embed_model):
    base_query_engine = query_engine_corrected_my_method(index, llm, embed_model)
    return TransformQueryEngine(base_query_engine, HyDEQueryTransform(llm=llm, include_original=True))

def query_engine_mmr(index, llm, embed_model):
    return index.as_query_engine(llm=llm,
          text_qa_template=TEXT_QA_TEMPLATE,
          similarity_top_k=3,
          embed_model=embed_model,
          vector_store_query_mode="mmr")

def query_engine_mmr_hyde(index, llm, embed_model):
    base_query_engine = query_engine_mmr(index, llm, embed_model)
    return TransformQueryEngine(base_query_engine, HyDEQueryTransform(llm=llm, include_original=True))
