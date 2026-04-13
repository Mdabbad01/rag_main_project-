from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_documents_into_chunks(documents, chunk_size=500, chunk_overlap=100):
    """
    Split documents into smaller chunks while preserving metadata.
    Returns a list of chunked LangChain Document objects.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks = text_splitter.split_documents(documents)

    # Add chunk IDs for easier debugging later
    for idx, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = idx

    return chunks