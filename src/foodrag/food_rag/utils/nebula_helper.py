import re
from llama_index.core import KnowledgeGraphIndex
from llama_index.core import StorageContext
from llama_index.core import load_index_from_storage

from foodrag.utils import load_util


def _import_nebula_graph_store():
    """Lazily import the optional NebulaGraph store.

    The Nebula path is a legacy/optional toggle (`pip install foodrag[nebula]`).
    Keeping the import lazy lets the pure helpers in this module (e.g.
    ``clean_nebula_entities``) be used without the optional dependency.
    """
    try:
        from llama_index.graph_stores.nebula import NebulaGraphStore
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "NebulaGraph support is optional. Install it with: "
            "uv sync --extra nebula  (or pip install 'foodrag[nebula]')."
        ) from exc
    return NebulaGraphStore


def clean_nebula_entities(text):
    # Define regex patterns
    null_rel_pattern = re.compile(r'\[(\w+):\{\1: null}]')
    empty_brackets_pattern = re.compile(r'\(\s*\)|\[\s*]|\{\s*}')

    # Split the input text into lines
    lines = text.split('\n')

    # Initialize a list to hold the cleaned lines
    cleaned_lines = []

    for line in lines:
        # Remove patterns matching the regex
        cleaned_line = re.sub(empty_brackets_pattern, '', line)

        # Substitute the matched patterns in the text
        cleaned_line = null_rel_pattern.sub(_replace_relation, cleaned_line)

        # Append the cleaned line if it's not empty
        if cleaned_line.strip():
            cleaned_lines.append(cleaned_line)

    # if cleaned_lines has only one element, return the element
    if len(cleaned_lines) == 1:
        return cleaned_lines[0]
    else:
        return cleaned_lines


def _replace_relation(match):
    relation = match.group(1)
    return f'[{relation}]'


def load_or_create_index(data_path, space_name, edge_types, rel_prop_names, tags, documents=[], include_embeddings=True,
                         verbose=False):
    NebulaGraphStore = _import_nebula_graph_store()
    graph_store = NebulaGraphStore(
        space_name=space_name,
        edge_types=edge_types,
        rel_prop_names=rel_prop_names,
        tags=tags,
    )

    load_util.ensure_dir(data_path)

    try:
        print('Loading existing space...')
        storage_context = StorageContext.from_defaults(persist_dir=data_path, graph_store=graph_store)
        kg_index = load_index_from_storage(
            storage_context=storage_context,
            max_triplets_per_chunk=15,
            space_name=space_name,
            edge_types=edge_types,
            rel_prop_names=rel_prop_names,
            tags=tags,
            verbose=verbose,
            include_embeddings=include_embeddings,
        )
    except Exception:
        print('Specified space not found! Creating new space...')
        storage_context = StorageContext.from_defaults(graph_store=graph_store)
        kg_index = KnowledgeGraphIndex.from_documents(
            # can use KnowledgeGraphIndex.from_documents only with the first types in entities and edges
            documents=documents,
            storage_context=storage_context,
            max_triplets_per_chunk=15,
            space_name=space_name,
            edge_types=edge_types,
            rel_prop_names=rel_prop_names,
            tags=tags,
            verbose=verbose, # not present in base .from_documents(), passed as kwargs --> might not be necessary
            show_progress=verbose,
            include_embeddings=include_embeddings,
        )

        print('Persisting the space...')
        kg_index.storage_context.persist(persist_dir=data_path)

    return kg_index


def load_index(data_path, space_name, edge_types, rel_prop_names, tags, include_embeddings=True, verbose=False):
    NebulaGraphStore = _import_nebula_graph_store()
    graph_store = NebulaGraphStore(
        space_name=space_name,
        edge_types=edge_types,
        rel_prop_names=rel_prop_names,
        tags=tags,
    )

    load_util.ensure_dir(data_path)

    try:
        print('Loading existing space...')
        storage_context = StorageContext.from_defaults(persist_dir=data_path, graph_store=graph_store)
        kg_index = load_index_from_storage(
            storage_context=storage_context,
            max_triplets_per_chunk=15,
            space_name=space_name,
            edge_types=edge_types,
            rel_prop_names=rel_prop_names,
            tags=tags,
            verbose=verbose,
            include_embeddings=include_embeddings,
        )
    except Exception:
        print('Specified space could not be loaded')

    return kg_index
