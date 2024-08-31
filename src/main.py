"""main.py"""
from pathlib import Path

from langchain_community.graphs import Neo4jGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_openai import ChatOpenAI

from src.constants import (
    NEO4J_PASSWORD,
    NEO4J_URI,
    NEO4J_USERNAME,
)
from src.text_process import doc_splitter


def main() -> None:
    """Main function"""
    graph = Neo4jGraph(
        NEO4J_URI,
        NEO4J_USERNAME,
        NEO4J_PASSWORD,
    )

    doc_path = Path("./docs/attack.txt")
    texts = doc_splitter(doc_path=doc_path)

    llm = ChatOpenAI(temperature=0, model="gpt-4o")

    llm_transformer = LLMGraphTransformer(llm=llm)
    graph_documents = llm_transformer.convert_to_graph_documents(texts)
    graph.add_graph_documents(graph_documents)


if __name__ == "__main__":
    main()
