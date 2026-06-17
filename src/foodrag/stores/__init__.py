"""Pluggable backing-store providers (vector store now; graph stores via helpers)."""
from foodrag.stores.vector import (
    PgVectorProvider,
    VectorStoreProvider,
    available_providers,
    get_vector_store_provider,
    register_provider,
)

__all__ = [
    "VectorStoreProvider",
    "PgVectorProvider",
    "get_vector_store_provider",
    "available_providers",
    "register_provider",
]
