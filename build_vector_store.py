from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.extractors import TitleExtractor
from llama_index.core.ingestion import IngestionPipeline
import chromadb
from llama_index.core import StorageContext
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from datasets import load_dataset
from tqdm import tqdm
import os
import nest_asyncio
nest_asyncio.apply()


class VectorStoreBuilder:
    def __init__(self, model_name='gpt-4o', text_directory='./data/text', chroma_db_path='./data/chroma_db', testing=False):
        self.setup(model_name) # self.llm, self.embed_model
        self.nodes = None
        self.text_directory = text_directory
        self.chroma_db_path = chroma_db_path
        self.chroma_collection = None
        self.testing = testing

    def setup(self, model_name):
        if os.getenv("OPENAI_API_KEY") is None:
            raise ValueError("OPENAI_API_KEY is not set")
        if os.getenv("OPENAI_API_BASE") is None:
            print("Using OpenAI API")
            self.llm = OpenAI(temperature=0, model=model_name, api_key=os.getenv("OPENAI_API_KEY"))
            self.embed_model = OpenAIEmbedding(model="text-embedding-3-large", api_key=os.getenv("OPENAI_API_KEY"))
        else:
            self.llm = OpenAI(temperature=0, model=model_name, api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_API_BASE"))
            self.embed_model = OpenAIEmbedding(model="text-embedding-3-large", api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_API_BASE"))


    def download_dataset(self):
        text_directory = './data/text'
        if not os.path.exists(text_directory):
            print("Downloading dataset...")
            os.makedirs(text_directory, exist_ok=True)

            dataset = load_dataset("jamescalam/ai-arxiv", split="train")

            count = 0
            for item in tqdm(dataset):
                with open(f"{text_directory}/{item['title'].replace('/', '_')}.txt", "w") as fp:
                    fp.write(item["content"])
                if self.testing:
                    if count > 10:
                        break
                    count += 1
        else:
            print("Dataset already downloaded")


    def _create_nodes(self):
        # create the pipeline with transformations
        pipeline = IngestionPipeline(
            transformations=[
                SentenceSplitter(chunk_size=2000, chunk_overlap=200),
                TitleExtractor(),
                self.embed_model,
            ],
        )

        # Load and process documents in batches
        documents = SimpleDirectoryReader(self.text_directory).load_data()
        batch_size = 40  # Process 50 documents at a time
        all_nodes = []
        
        for i in tqdm(range(0, len(documents), batch_size), desc="Processing document batches"):
            batch = documents[i:i + batch_size]
            batch_nodes = pipeline.run(documents=batch, num_workers=10)  # Reduced number of workers
            all_nodes.extend(batch_nodes)
            
        return all_nodes


    def create_vector_store(self):
        db = chromadb.PersistentClient(path="./data/chroma_db", settings=chromadb.config.Settings(allow_reset=True, is_persistent=True))
        self.chroma_collection = db.get_or_create_collection("ai_arxiv_full")
        vector_store = ChromaVectorStore(chroma_collection=self.chroma_collection)

        if self.chroma_collection.count() == 0:
            print("Creating vector store...")
            storage_context = StorageContext.from_defaults(vector_store=vector_store)

            nodes = self._create_nodes()

            return VectorStoreIndex(nodes, storage_context=storage_context)
        else:
            print("Loading vector store...")
            return VectorStoreIndex.from_vector_store(vector_store, embed_model=self.embed_model)


def main():
    vector_store_builder = VectorStoreBuilder(model_name='gpt-4o', testing=True)
    vector_store_builder.download_dataset()
    index = vector_store_builder.create_vector_store()

    query_engine = index.as_query_engine(embed_model=vector_store_builder.embed_model)
    response = query_engine.query("What is the main idea of the paper about red-teaming?")
    print(response)


if __name__ == "__main__":
    main()