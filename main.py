from build_vector_store import VectorStoreBuilder
from retrieve import get_retrieved_nodes, pretty_print, visualize_retrieved_nodes

def main():
    vector_store_builder = VectorStoreBuilder(model_name='gpt-4o')
    vector_store_builder.download_dataset()
    index = vector_store_builder.create_vector_store()

    query_engine = index.as_query_engine()
    response = query_engine.query("What is the main idea of the paper about red-teaming?")
    print(response)


if __name__ == "__main__":
    main()