import uuid
from deepagents.backends import StateBackend
from langchain.tools import tool
from indexing import vector_store

backend = StateBackend()

@tool(parse_docstring=True)
def search_documentation(query:str) -> str:
    """Search langchain documents and save matching chunks to the agent filesystem
    
    Args:
        query:Natural Language Search Query

    Returns:
        File paths wwhere retrieved chunks were saved under /retrieved/.
    """
    retrieved_docs = vector_store.similarity_search(query,k=4)
    batch_id=uuid.uuid4().hex[:8]
    uploads: list[tuple[str,bytes]] = []
    saved_paths:list[str] = []

    for index,doc in enumerate(retrieved_docs, start=1):
        path=f"/retrieved/{batch_id}/chunk_{index}.md"
        content = (
            f"# Source: {doc.metadata.get('source','unknown')}\n\n"
            f"{doc.page_content}"
        )
        uploads.append((path,content.encode("utf-8")))
        saved_paths.append(path)

    backend.upload_files(uploads)
    return(
        f"saved {len(saved_paths)} document chunks"
        + "\n".join(saved_paths)
    )
