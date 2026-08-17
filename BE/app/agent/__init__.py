"""Agent package public API. All access is lazy-initialized via PEP 562."""

from app.agent.graph import get_compiled_graph
from app.services.registry import get_services


def __getattr__(name):
    if name == "compiled_graph":
        return get_compiled_graph()
    if name == "graph_indexing_service":
        return get_services().graph_indexing_service
    if name == "community_service":
        return get_services().community_service
    if name == "global_search_service":
        return get_services().global_search_service
    if name == "graph_query_service":
        return get_services().graph_query_service
    if name == "ark_query_service":
        return get_services().ark_query_service
    if name == "llm":
        return get_services().llm
    raise AttributeError(f"module 'app.agent' has no attribute '{name}'")
