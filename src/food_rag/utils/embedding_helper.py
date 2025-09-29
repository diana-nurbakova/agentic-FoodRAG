from collections import defaultdict

import math
import os
import pickle
import time
from llama_index.core import Settings
from llama_index.core.node_parser import TokenTextSplitter
from openai import RateLimitError
from tqdm.notebook import tqdm
from transformers import AutoModel, AutoTokenizer
from typing import List

from src.utils import load_util

QA_SUB_PATH = '/qa_documents/'

# in seconds --> max. 1 hour cooldown for a single request (in total as it is increased by 20% each time the real wait
# time is way higher depending on the starting value)
MAX_COOLDOWN = 60 * 60 * 1


def get_tokenizer_from_embed_model(model_name: str):
    return AutoTokenizer.from_pretrained(model_name)


def get_encoder_from_embed_model(model_name: str):
    return AutoModel.from_pretrained(model_name)


class CallableTokenizer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

        # check if method tokenize exists
        if not hasattr(self.tokenizer, 'tokenize'):
            raise ValueError('Tokenizer must have a method tokenize!')

    def __call__(self, item):
        return self.tokenizer.tokenize(item)


def compute_embeddings(documents, embed_model, batch_size=10000, chunk_size=512, chunk_overlap=10, path=None,
                       minimize_memory=False, embedd_query=False):
    """
    Tokenizes documents and adds embeddings to corresponding nodes (=tokenized docs) in batches. Optionally, save the
    results after every batch. Automatically skips nodes that already have an embedding and loads previous batches if
    save_path is provided.
    Note: If the embedding fails for a document, the document is split into two parts and the embedding is tried again.

    :param documents: list of documents to process
    :param embed_model: model used to generate embeddings
    :param batch_size: optional, the number of nodes to process in each batch (default: all at once)
    :param chunk_size: optional, the number of tokens in each chunk (default: 512). Ignored if a tokenizer is provided.
    :param chunk_overlap: optional, the number of overlapping tokens between chunks (default: 10)
    :param path: optional, path to save the nodes after every batch, and load existing batches if available
    :param minimize_memory: optional, if True, the nodes are not stored in memory after processing (default: False)
    :param embedd_query: If True, the query is embedded instead of the documents (default: False) (another embedding
    method is used then).

    :return: a dict of the documents and their corresponding nodes
    """
    splitter, small_splitter, inferred_tokenizer = infer_splitter(embed_model, chunk_size, chunk_overlap)

    # Set default batch size if not provided
    if batch_size is None:
        batch_size = len(documents)  # Process all documents at once if no batch size is given

    # Load previously saved batches if a save_path is provided
    if path:
        if minimize_memory:
            print("Minimizing memory by only loading document hashes.")
            doc_nodes_dict, loaded_batch_names = load_existing_batch_hashes(path)
        else:
            doc_nodes_dict, loaded_batch_names = load_existing_batches(path)

        # Necessary to compare docs as documents are not hashable
        doc_hashes = [doc.hash for doc in documents]

        # Only keep loaded documents which are also present in the documents param
        doc_nodes_dict = {doc_hash: nodes for doc_hash, nodes in doc_nodes_dict.items() if doc_hash in doc_hashes}
    else:
        doc_nodes_dict = {}
        loaded_batch_names = []

    # Filter documents that do not have an embedding
    unprocessed_docs = [doc for doc in documents if doc.hash not in doc_nodes_dict]

    # Remove the stored documents from memory if minimize_memory is set
    del documents

    # Process nodes in batches
    for i in tqdm(range(0, len(unprocessed_docs), batch_size), desc="Processing batches"):
        batch = unprocessed_docs[i:i + batch_size]
        print(f"Processing batch {i // batch_size + 1} of {len(unprocessed_docs) // batch_size + 1}")

        # Generate embeddings for the current batch
        try:
            # 1. Tokenize the documents --> get nodes
            # Note: Tokenizer might print warnings if the input is too long --> does not matter as we split the nodes then
            nodes = splitter.get_nodes_from_documents(batch, show_progress=False)

            # 2. Embed the nodes
            if not embedd_query:  # default, just text is embedded
                embeddings = embed_model.get_text_embedding_batch(
                    [node.get_content(metadata_mode="embed") for node in nodes],
                    show_progress=True
                )
            else:  # query is embedded
                embeddings = [embed_model.get_query_embedding(node) for node in nodes]

            # 3.1 Assign embeddings to nodes
            _ = [node.__setattr__('embedding', embedding) for node, embedding in zip(nodes, embeddings)]

            # 3.2 Add the processed nodes to the respective documents
            batch_doc_nodes_dict = defaultdict(list)
            for node in nodes:
                respective_doc = next(doc for doc in batch if doc.id_ == node.ref_doc_id)
                batch_doc_nodes_dict[respective_doc.hash].append(node)

        except Exception:
            print(f"Failed embedding batch {i // batch_size + 1}. Trying to embed individual docs instead.")
            time.sleep(60)  # Sleep for a minute as the API might be rate limited

            # Embedd entities for each individual document
            batch_doc_nodes_dict = {}
            for doc in tqdm(batch, desc="Embedding individual docs"):
                _embedd_single_doc(doc, embed_model, batch_doc_nodes_dict, inferred_tokenizer, small_splitter, splitter)

        # Save the current batch if a save_path is provided
        if path:
            load_util.ensure_dir(path)
            loaded_batch_names = save_batch(dict(batch_doc_nodes_dict), path, loaded_batch_names,
                                            batch_index=i // batch_size + 1)

        # 3.3 Add result to doc_node_dict
        if not minimize_memory:
            doc_nodes_dict.update(dict(batch_doc_nodes_dict))

    return doc_nodes_dict


def compute_embedding(document, embed_model, splitter, small_splitter):
    """
    Tokenizes a document and adds embeddings to corresponding nodes. If the embedding fails for a document, the document
     is split by the back-up splitter (small_splitter) and the embedding is tried again.
    :param document: The document to process
    :param embed_model: The model used to generate embeddings
    :param splitter: The splitter used to tokenize the document
    :param small_splitter: The splitter used to tokenize the document in case the original tokenizer fails
    :return: A list of nodes with embeddings
    """
    doc_nodes_dict = {}
    _embedd_single_doc(document, embed_model, doc_nodes_dict, True, small_splitter, splitter)
    return doc_nodes_dict[document.hash]


def _embedd_single_doc(doc, embed_model, batch_doc_nodes_dict, inferred_tokenizer, small_splitter, splitter,
                       current_cooldown=60):
    try:
        # 1. Tokenize the document --> get nodes
        nodes = splitter.get_nodes_from_documents([doc], show_progress=False)

        # 2. Embed the nodes
        embeddings = embed_model.get_text_embedding_batch(
            [node.get_content(metadata_mode="embed") for node in nodes],
            show_progress=False
        )

        # 3.1 Assign embeddings to nodes
        _ = [node.__setattr__('embedding', embedding) for node, embedding in zip(nodes, embeddings)]

        # 3.2 Add the processed nodes to the document
        batch_doc_nodes_dict[doc.hash] = nodes
    except RateLimitError as e:
        current_cooldown_in_minutes = current_cooldown // 60
        current_cooldown_left_seconds = current_cooldown % 60
        if current_cooldown < MAX_COOLDOWN:
            print(f"Rate limit exceeded. Sleeping for {current_cooldown_in_minutes} minutes and "
                  f"{current_cooldown_left_seconds} seconds and trying again. "
                  f"Detailed RateLimitError:\n{e}")
            time.sleep(current_cooldown)
            next_cooldown = math.ceil(current_cooldown * 1.2)  # Increase next possible cooldown by 20%

            # Retry embedding the document
            _embedd_single_doc(doc, embed_model, batch_doc_nodes_dict, inferred_tokenizer, small_splitter, splitter,
                               current_cooldown=next_cooldown)
        else:
            print(f"Rate limit exceeded. Sleep time of {current_cooldown} exceeds the maximal sleep time of "
                  f"{MAX_COOLDOWN}. Skipping document:\n{doc}")

    except Exception as e:
        # Split the node into smaller parts and try to embed them
        if inferred_tokenizer:
            print(f"Failed embedding document although the original tokenizer was used. Splitting into"
                  f" smaller parts and trying again. Detailed Exception:\n{e}")
        else:
            print(f"Failed embedding document. Splitting into smaller parts and trying again. Detailed Exception:\n{e}")

        time.sleep(5)  # Sleep for 5 seconds to avoid rate limit just in case

        if small_splitter is None:
            print("Small Splitter not available. Skipping document.")
            time.sleep(5)  # Sleep for 5 seconds to avoid rate limit just in case
        else:
            try:
                # 1. Tokenize the document --> get nodes
                nodes = small_splitter.get_nodes_from_documents([doc], show_progress=False)

                # 2. Embed the nodes
                embeddings = embed_model.get_text_embedding_batch(
                    [node.get_content(metadata_mode="embed") for node in nodes],
                    show_progress=False
                )

                # 3.1 Assign embeddings to nodes
                _ = [node.__setattr__('embedding', embedding) for node, embedding in zip(nodes, embeddings)]

                # 3.2 Add the processed nodes to the document
                batch_doc_nodes_dict[doc.hash] = nodes
            except Exception as e:
                print(f"Also failed when splitting document in half. Skipping document:\n{doc}\nException:\n{e}")
                time.sleep(5)  # Sleep for 5 seconds to avoid rate limit just in case


def infer_splitter(embed_model, chunk_size=512, chunk_overlap=10):
    """
    Infer the splitter to use based on the embed model. If the tokenizer can be inferred from the embed model, the
    chunk_size parameter is ignored.

    :param embed_model: The embed model to infer the tokenizer from
    :param chunk_size: The size of the chunks to split the documents into
    :param chunk_overlap: The number of overlapping tokens between chunks
    :return: The splitter to use, a back-up splitter (splitting the document into 2 parts), and the inferred tokenizer
    """
    try:
        tokenizer = get_tokenizer_from_embed_model(embed_model.model_name)
        splitter = TokenTextSplitter(tokenizer=CallableTokenizer(tokenizer),
                                     chunk_size=tokenizer.max_len_single_sentence, chunk_overlap=chunk_overlap)
        small_splitter = TokenTextSplitter(tokenizer=CallableTokenizer(tokenizer),
                                           chunk_size=tokenizer.max_len_single_sentence // 2,
                                           chunk_overlap=chunk_overlap)
        inferred_tokenizer = True
    except OSError:
        try:  # try to get tokenizer from model name without "openai/" prefix
            short_name = embed_model.model_name.removeprefix("openai/")
            tokenizer = get_tokenizer_from_embed_model(short_name)

            splitter = TokenTextSplitter(
                tokenizer=CallableTokenizer(tokenizer),
                chunk_size=tokenizer.max_len_single_sentence,
                chunk_overlap=chunk_overlap)
            small_splitter = TokenTextSplitter(
                tokenizer=CallableTokenizer(tokenizer),
                chunk_size=tokenizer.max_len_single_sentence // 2,
                chunk_overlap=chunk_overlap)

            inferred_tokenizer = True
        except OSError:
            splitter = TokenTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separator=" ",
            )
            small_splitter = TokenTextSplitter(
                chunk_size=chunk_size // 2,
                chunk_overlap=chunk_overlap,
                separator=" ",
            )

            inferred_tokenizer = False

    if inferred_tokenizer:
        print(f"Using tokenizer from embed model: {embed_model.model_name}. Ignoring chunk_size parameter.")
    else:
        print(f"Using custom tokenizer. Chunk size: {chunk_size}")

    return splitter, small_splitter, inferred_tokenizer


def load_existing_batches(save_path, nodes_only=False, max_batches=None, verbose=False):
    """
    Load previously saved batches from the save path.

    :param save_path: path to load the previously saved batches from
    :param nodes_only: if True, only the nodes are loaded, otherwise the full document-node dict is loaded.
    Default: False
    :param max_batches: maximum number of batches to load. If None, all batches are loaded.
    :param verbose: if True, print additional information. Default: False. The loading process is always displayed.

    :return: A dict of document-nodes (or just the nodes) that have been loaded and a list of the names of the loaded
     batch files.
    """
    loaded_dict = defaultdict(list)
    loaded_batch_names = []
    if not os.path.exists(save_path):
        load_util.ensure_dir(save_path)
    else:
        batch_counter = 0

        # Iterate through files in the save_path directory
        for file_name in tqdm(os.listdir(save_path), desc="Loading batches"):
            if max_batches is None or batch_counter < max_batches:
                if file_name.startswith("batch_") and file_name.endswith(".pkl"):
                    full_path = os.path.join(save_path, file_name)
                    with open(full_path, 'rb') as f:
                        batch_dict = pickle.load(f)
                        _ = {loaded_dict[doc_hash].extend(nodes) for doc_hash, nodes in batch_dict.items()}
                        loaded_batch_names.append(file_name)
                        batch_counter += 1
                    if verbose:
                        print(f"Loaded {file_name} with {len(batch_dict.keys())} documents.")
                else:
                    if verbose:
                        print(f"Skipping file {file_name} as it does not match the batch naming convention.")

    if nodes_only:
        nodes = [node for doc_nodes in loaded_dict.values() for node in doc_nodes]
        return nodes, loaded_batch_names
    else:
        return dict(loaded_dict), loaded_batch_names


def load_existing_batch_hashes(save_path):
    """
    Load previously saved hashes of batches from the save path.

    :param nodes: list of nodes to check and load
    :param save_path: path to load the previously saved batches from

    :return: A dict of document_hash-nodes that have been loaded and a list of the names of the loaded batch files.
    """
    loaded_dict = defaultdict(list)
    loaded_batch_names = []
    if not os.path.exists(save_path):
        load_util.ensure_dir(save_path)
    else:
        for file_name in tqdm(os.listdir(save_path),
                              desc="Loading batch hashes"):  # Iterate through files in the save_path directory
            if file_name.startswith("batch_") and file_name.endswith(".pkl"):
                full_path = os.path.join(save_path, file_name)
                with open(full_path, 'rb') as f:
                    batch_dict = pickle.load(f)
                    _ = {loaded_dict[doc_hash].extend([]) for doc_hash, nodes in batch_dict.items()}
                    loaded_batch_names.append(file_name)
                print(f"Loaded {file_name} with {len(batch_dict.keys())} documents.")
            else:
                print(f"Skipping file {file_name} as it does not match the batch naming convention.")

    return dict(loaded_dict), loaded_batch_names


def save_batch(doc_node_dict, save_path, loaded_batch_names, batch_index):
    """
    Save the current batch of document-nodes-dicts to the specified path. Updates the list of loaded batch names.

    :param doc_node_dict: document-node-dict in the current batch
    :param save_path: path where the batch is saved
    :param loaded_batch_names: list of names of the already loaded batch files
    :param batch_index: index of the current batch

    :return: updated list of loaded batch names
    """
    file_name = f"batch_{batch_index}.pkl"

    if file_name in loaded_batch_names:
        print(f"Batch {batch_index} already present. Storing this batch with a different name.")

    while file_name in loaded_batch_names:
        batch_index += 1
        file_name = f"batch_{batch_index}.pkl"

    full_path = os.path.join(save_path, file_name)

    # Save the batch to a file
    with open(full_path, 'wb') as f:
        pickle.dump(doc_node_dict, f)

    loaded_batch_names.append(file_name)
    print(f"Batch {batch_index} saved to {full_path}")

    return loaded_batch_names


def check_batches(recipe_documents, doc_nodes_dict):
    # Check if all recipe documents are loaded
    print("All recipe documents are loaded." if all(doc.hash in doc_nodes_dict for doc in recipe_documents)
          else f"At least document {next(doc.hash for doc in recipe_documents if doc.hash not in doc_nodes_dict)} "
               f"not found in the loaded batches.")


def extend_doc_embeddings(recipe_docs, recipe_loaded_doc_dict):
    """
    Extend the embeddings of the recipe documents with the embeddings from the loaded recipe documents.

    :param recipe_docs: list of recipe documents
    :param recipe_loaded_doc_dict: dict of loaded recipe documents and their embeddings

    :return: list of cleaned recipe documents with extended embeddings
    """
    # hash --> nodes to recipe_id --> nodes
    recipe_loaded_doc_dict = {nodes[0].metadata['recipe_id']:
                                  nodes for nodes in recipe_loaded_doc_dict.values() if len(nodes) > 0}

    for doc in recipe_docs:
        if doc.metadata['recipe_id'] in recipe_loaded_doc_dict:
            node_embeddings = [node.embedding for node in recipe_loaded_doc_dict[doc.metadata['recipe_id']]]
            try:
                doc.embedding = node_embeddings
            except ValueError as e:
                # Embeddings might be split into multiple parts as the doc was too large --> take the first embedding
                doc.embedding = node_embeddings[0]

    return recipe_docs


def embed_query(query: str, verbose=True) -> List[float]:
    """
    Embed a single text query using the embedding client with retry logic.

    :param str query: The input query to embed.
    :param bool verbose: If True, print messages. Default is True.
    :return: A list of floats representing the embedding vector.
    :rtype: List[float]
    """
    sleep_time = 1.0
    while True:
        try:
            return Settings.embed_model.get_query_embedding(query)
        except Exception:
            if verbose:
                print(f"Embedding failed. Retrying in {sleep_time} seconds...")
            time.sleep(sleep_time)
            sleep_time *= 1.5
