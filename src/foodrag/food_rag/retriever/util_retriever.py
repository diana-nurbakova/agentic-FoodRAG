from collections import Counter, defaultdict

import math
import numpy as np
import random
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from llama_index.core import QueryBundle
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.schema import NodeWithScore, Document
from typing import List


class RandomRetriever(BaseRetriever):
    SCORE_METHODS = ["random", "1"]

    def __init__(self, nodes, k=5, score_method="1"):
        """Initialize the random retriever.
        :param nodes (list): List of nodes to sample from.
        :param k (int): Number of nodes to retrieve.
        :param score_method (str): Method to score the nodes. Options are "random" or "1". Default is 1.
        """
        if score_method not in self.SCORE_METHODS:
            raise ValueError(f"Invalid score method. Choose from {self.SCORE_METHODS}.")

        super().__init__()
        self.nodes = nodes
        self.k = k
        self.score_method = score_method

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        sampled_nodes = random.sample(self.nodes, min(self.k, len(self.nodes)))
        return [
            NodeWithScore(node=n, score=random.random() if self.score_method == "random" else 1)
            for n in sampled_nodes
        ]


class Doc2VecRetriever(BaseRetriever):
    def __init__(self, nodes, vector_size=50, window=2, min_count=1, epochs=20, similarity_top_k=5):
        super().__init__()
        self.docs = get_unique_typed_docs(nodes)
        self.similarity_top_k = similarity_top_k

        self.model = Doc2Vec(
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            epochs=epochs
        )

        self.doc_vectors = {}  # node_id -> np.array
        self._train_model()

    def _train_model(self):
        documents = []
        for doc in self.docs:
            tokens = doc.text.lower().split()
            documents.append(TaggedDocument(tokens, [doc.id_]))

        self.model.build_vocab(documents)
        self.model.train(documents, total_examples=len(documents), epochs=self.model.epochs)

        for doc in self.docs:
            tokens = doc.text.lower().split()
            self.doc_vectors[doc.id_] = self.model.infer_vector(tokens)

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        query_vec = self.model.infer_vector(query_bundle.query_str.lower().split())

        scored = []
        for doc in self.docs:
            doc_vec = self.doc_vectors[doc.id_]
            num = np.dot(query_vec, doc_vec)
            denom = np.linalg.norm(query_vec) * np.linalg.norm(doc_vec)
            sim = num / denom if denom else 0
            scored.append((doc, sim))

        top_nodes = sorted(scored, key=lambda x: x[1], reverse=True)[:self.similarity_top_k]
        return [NodeWithScore(node=n, score=s) for n, s in top_nodes]


class BoWCosineRetriever(BaseRetriever):
    def __init__(self, nodes, similarity_top_k=5):
        super().__init__()
        self.docs = get_unique_typed_docs(nodes)
        self.similarity_top_k = similarity_top_k
        self.vocab = []
        self.doc_vectors = {}  # node_id -> list
        self._build_vocab_and_vectors()

    def _build_vocab_and_vectors(self):
        all_texts = [doc.text.lower().split() for doc in self.docs]
        self.vocab = sorted(set(word for text in all_texts for word in text))

        for doc in self.docs:
            self.doc_vectors[doc.id_] = self._bow_vector(doc.text)

    def _bow_vector(self, text):
        words = text.lower().split()
        counts = Counter(words)
        return [counts.get(w, 0) for w in self.vocab]

    def _cosine(self, vec1, vec2):
        dot = sum(a * b for a, b in zip(vec1, vec2))
        mag1 = math.sqrt(sum(a * a for a in vec1))
        mag2 = math.sqrt(sum(b * b for b in vec2))
        return dot / (mag1 * mag2) if mag1 and mag2 else 0

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]: # TODO DOES NOT TERMINATE OR TAKE VERY LONG
        query_vec = self._bow_vector(query_bundle.query_str)

        scored = []
        for doc in self.docs:
            doc_vec = self.doc_vectors[doc.id_]
            score = self._cosine(query_vec, doc_vec)
            scored.append((doc, score))

        top_nodes = sorted(scored, key=lambda x: x[1], reverse=True)[:self.similarity_top_k]
        return [NodeWithScore(node=n, score=s) for n, s in top_nodes]


def get_unique_typed_docs(nodes: List[NodeWithScore]) -> List[Document]:
    """
    Get unique typed nodes.
    :param nodes: List of nodes to filter.
    :return: List of unique nodes.
    """
    unique_typed_docs = defaultdict(list)
    typed_nodes = defaultdict(list)

    # Group nodes by their type
    [typed_nodes[node.metadata.get('type', 'other')].append(node.source_node) for node in nodes]

    for node_type, node_set in typed_nodes.items():
        added_ids = set()
        for node in node_set:
            if node_type in ['ingredient', 'food_on_product']:
                doc = Document(id_=str(hash(node.metadata['name'])), text=node.metadata['name'], metadata=node.metadata)
            elif node_type == 'food_product':
                doc = Document(id_=str(hash(node.metadata['name'])), text=node.metadata['name'].lower(),
                               metadata=node.metadata)
            else:  # invalid type, skip
                print(f"Invalid node type: {node_type}")
                continue

            # Add the doc to the set of unique nodes
            if doc.id_ not in added_ids:
                added_ids.add(doc.id_)
                unique_typed_docs[node_type].append(doc)

    # Append the unique nodes to the list
    docs = []
    for node_type, doc_list in unique_typed_docs.items():
        docs.extend(doc_list)

    return docs
