from src.doctract.rag.api.models import load_huggingface_embedding_model, \
    load_llama_cpp_language_model
from src.doctract.rag.loader import load_pdf_document_from_data_dir, \
    split_document_into_text_chunks
from src.doctract.rag.indexer import create_text_nodes_with_metadata, \
    embed_text_nodes_with_model
from src.doctract.rag.retriever import VectorDatabaseRetriever
from src.doctract.rag.api.psql import initialize_pgvector_store
from llama_index.core.query_engine import RetrieverQueryEngine


def main():
    vector_store = initialize_pgvector_store()
    embedding_model = load_huggingface_embedding_model()
    llm_model = load_llama_cpp_language_model()
    documents = load_pdf_document_from_data_dir(filename="llama2.pdf")
    text_chunks, doc_idxs = split_document_into_text_chunks(documents=documents)
    nodes = create_text_nodes_with_metadata(text_chunks=text_chunks,
                                            source_document_indices=doc_idxs,
                                            original_documents=documents)
    nodes = embed_text_nodes_with_model(
        text_nodes=nodes,
        embedding_model=embedding_model
    )
    vector_store.add(nodes)
    query_str = "Can you tell me about the key concepts for safety fine tuning"
    retriever = VectorDatabaseRetriever(
        vector_store_backend=vector_store,
        embedding_model=embedding_model
    )
    # retriever._retrieve(query_str=query_str)
    query_engine = RetrieverQueryEngine.from_args(retriever, llm=llm_model)
    response = query_engine.query(query_str)
    print(str(response))

    print(response.source_nodes[0].get_content())


if __name__ == "__main__":
    main()