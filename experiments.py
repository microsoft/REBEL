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

RELEVANCE_ONLY_REBEL_META_PROMPT_TMPL = PromptTemplate("""You are a prompt generator. You will receive only a user's query as input. Your task is to produce a final prompt that instructs a large-language model re-ranker to process a set of candidate documents solely based on their relevance to the user's query. The re-ranker will see the documents and the query in the following format:
    
A list of documents is shown below. Each document has a number and a summary. The summaries may indicate the source type, publication date, or other contextual details. After listing all documents, the user's query is presented on a single line labeled "Question:".
    
For example:
    
Document 1:
<summary of document 1>
    
Document 2:
<summary of document 2>
    
...
    
Document N:
<summary of document N>
    
Question: <user's query>
    
Your final prompt must instruct the re-ranker as follows:
    
1. For each document, assign a Relevance score from 0 to 10, where:
   - A score of 10 means the document directly and comprehensively addresses the user's query.
   - A score of 0 means the document is completely off-topic.
    
2. Immediately discard any document with a Relevance score lower than 3.
    
3. For the remaining documents, use the Relevance score as the sole metric to compute a Final Score (i.e. Final Score = Relevance).
    
4. Sort the documents in descending order of their Final Score.
    
5. If two documents have the same Final Score, maintain their original order.
    
6. If no document qualifies (i.e. all have Relevance < 3), output nothing.
    
7. Output only the final ranked list, with each line in the following format (and no extra commentary):
Doc: [document number], Relevance: [Final Score]

    
At the end of your prompt, ALWAYS include the following exact example to illustrate the expected format:
    
"Example format: \n"
"Document 1:\n<summary of document 1>\n\n"
"Document 2:\n<summary of document 2>\n\n"
"...\n\n"
"Document 10:\n<summary of document 10>\n\n"
"Question: <question>\n"
"Answer:\n"
"Doc: 9, Relevance: 7\n"
"Doc: 3, Relevance: 4\n"
"Doc: 7, Relevance: 3\n\n"
"Let's try this now: \n\n"
"{context_str}\n"
"Question: {query_str}\n"
"Answer:\n"
    
Remember to include the user's query verbatim at the end of your prompt.
    
Now, here is the user's query:
{user_query}
""")


def query_engine_no_rerank(index, llm, embed_model):
    return index.as_query_engine(llm=llm,
          text_qa_template=TEXT_QA_TEMPLATE,
          similarity_top_k=3,
          embed_model=embed_model)

def query_engine_cohere_rerank(index, embed_model):
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

def query_engine_one_turn_rebel_rerank(index, llm, embed_model):
    llm_rerank = REBELRerank(
        one_turn=True,
        llm=llm,
        choice_batch_size=10,
        top_n=3)
    return index.as_query_engine(llm=llm,
          text_qa_template=TEXT_QA_TEMPLATE,
          similarity_top_k=10,
          node_postprocessors=[llm_rerank],
          embed_model=embed_model)

def query_engine_two_turn_rebel_rerank(index, llm, embed_model):
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

def query_engine_two_turn_relevance_only_rebel_rerank(index, llm, embed_model):
    rebel_rerank = REBELRerank(
        meta_prompt=RELEVANCE_ONLY_REBEL_META_PROMPT_TMPL,
        llm=llm,
        choice_batch_size=10,
        top_n=3)
    return index.as_query_engine(
        similarity_top_k=10,
        node_postprocessors=[rebel_rerank],
        text_qa_template=TEXT_QA_TEMPLATE,
        embed_model=embed_model)

def query_engine_hyde(index, llm, embed_model):
    base_query_engine = query_engine_no_rerank(index, llm, embed_model)
    return TransformQueryEngine(base_query_engine, HyDEQueryTransform(llm=llm, include_original=True))

def query_engine_mmr(index, llm, embed_model):
    return index.as_query_engine(llm=llm,
          text_qa_template=TEXT_QA_TEMPLATE,
          similarity_top_k=3,
          embed_model=embed_model,
          vector_store_query_mode="mmr")
