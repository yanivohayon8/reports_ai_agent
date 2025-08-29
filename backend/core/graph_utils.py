from typing import Optional


def get_graph_context(graph_indexer, query: str) -> str:
    """
    Safely query the provided graph_indexer and return a string context.
    Returns empty string on any error or when indexer is None.
    """
    if graph_indexer is None:
        return ""
    try:
        result = graph_indexer.retrieve(query)
        return str(result) if result is not None else ""
    except Exception:
        return ""


def augment_context_with_graph(graph_indexer, base_context: str, query: str) -> str:
    """
    Prepend graph-derived context to the base context if available.
    """
    graph_context = get_graph_context(graph_indexer, query)
    if not graph_context:
        return base_context
    if not base_context:
        return graph_context
    return f"{graph_context}\n\n{base_context}"


