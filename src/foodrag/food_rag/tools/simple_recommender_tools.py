from collections import Counter

import json
import math
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from llama_index.core import Response
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.notebook import tqdm
from typing import List
from rank_bm25 import BM25Okapi
from foodrag.food_rag.utils import embedding_helper


class BaseRecommender(ABC):
    """
    Abstract base class for all recommenders.
    """

    @abstractmethod
    def train(self, recipe_docs):
        """
        Train the model using recipe documents.

        :param recipe_docs: List of recipe documents, where each document has `text` and `metadata['recipe_id']`.
        """
        pass

    @abstractmethod
    def recommend_top1(self, query, valid_recipes_df) -> pd.DataFrame:
        """
        Recommend the top-1 recipe based on the given query.

        :param query: Query string to find the best matching recipe.
        :param valid_recipes_df: DataFrame containing a subset of recipes to recommend from.
        :return: Best matching recipe as df.
        """
        pass

    def recommend_top1_batch(self, queries, valid_recipes_df, only_print_progress=False) -> List[pd.DataFrame]:
        """
        Recommend the top-1 recipe for multiple queries using the given recommender.

        :param queries: List of query strings.
        :param valid_recipes_df: DataFrame containing valid recipes with 'recipe_id' as a key.
        :param only_print_progress: If True, only print the progress bar without a tqdm bar. Else, use tqdm for
         displaying the progress.
        :return: List of best matched recipe dfs.
        """
        results = []
        for i, query in tqdm(enumerate(queries), desc="Processing Queries", disable=only_print_progress):
            if only_print_progress:
                print(f"Processing query #{i+1}/{len(queries)}")

            best_match = self.recommend_top1(query, valid_recipes_df)
            results.append(best_match)
        return results


class TfidfRecommender(BaseRecommender):
    """
    1) Fit a TF-IDF vectorizer on recipe_docs (doc.text).
    2) Store each doc's TF-IDF vector: {recipe_id -> np.array}.
    3) For recommendation on valid_recipes_df:
       - Transform the query once.
       - Compare with the subset's vectors, pick top row.
    """

    def __init__(self):
        self.vectorizer = None
        self.recipe_vectors = {}  # {recipe_id -> np.array}
        self.is_trained = False

    def train(self, recipe_docs):
        # Build text corpus
        texts = [doc.text for doc in recipe_docs]
        self.vectorizer = TfidfVectorizer()
        tfidf_matrix = self.vectorizer.fit_transform(texts)  # shape (num_docs, dim)

        # Store each doc vector by recipe_id
        for i, doc in enumerate(recipe_docs):
            recipe_id = doc.metadata["recipe_id"]
            self.recipe_vectors[recipe_id] = tfidf_matrix[i].toarray()[0]

        self.is_trained = True

    def recommend_top1(self, query, valid_recipes_df) -> pd.DataFrame:
        if not self.is_trained:
            raise ValueError("TfidfRecommender not trained. Call train(...) first.")

        # Transform query once
        query_vec = self.vectorizer.transform([query])  # shape (1, dim)

        # Gather vectors for the subset
        subset_vectors = []
        for _, row in valid_recipes_df.iterrows():
            recipe_id = row["recipe_id"]
            subset_vectors.append(self.recipe_vectors[recipe_id])

        # Convert to matrix: (num_subset, dim)
        subset_matrix = np.vstack(subset_vectors)

        # Single cosine similarity call => shape (1, num_subset)
        sims = cosine_similarity(query_vec, subset_matrix)[0]
        best_index = np.argmax(sims)

        # Return that row
        return valid_recipes_df.iloc[best_index].to_frame()


class KeywordOverlapRecommender(BaseRecommender):
    """
    1) Pre-store each doc's set of words: {recipe_id -> set_of_words}.
    2) For recommendation on valid_recipes_df:
       - Build the query set once, pick top overlap from subset.
    """

    def __init__(self):
        self.recipe_wordsets = {}  # {recipe_id -> set_of_words}
        self.is_trained = False

    def train(self, recipe_docs):
        for doc in recipe_docs:
            recipe_id = doc.metadata["recipe_id"]
            words = set(doc.text.lower().split())
            self.recipe_wordsets[recipe_id] = words
        self.is_trained = True

    def recommend_top1(self, query, valid_recipes_df) -> pd.DataFrame:
        if not self.is_trained:
            raise ValueError("KeywordOverlapRecommender not trained. Call train(...) first.")

        query_words = set(query.lower().split())

        best_row = None
        best_score = -1

        for _, row in valid_recipes_df.iterrows():
            recipe_id = row["recipe_id"]
            recipe_words = self.recipe_wordsets[recipe_id]
            overlap = len(query_words.intersection(recipe_words))
            if overlap > best_score:
                best_score = overlap
                best_row = row

        return best_row.to_frame()


class JaccardRecommender(BaseRecommender):
    """
    1) Pre-store each doc's set of words: {recipe_id -> set_of_words}.
    2) For recommendation on valid_recipes_df:
       - Build the query set once, pick top Jaccard score.
    """

    def __init__(self):
        self.recipe_wordsets = {}  # {recipe_id -> set_of_words}
        self.is_trained = False

    def train(self, recipe_docs):
        for doc in recipe_docs:
            recipe_id = doc.metadata["recipe_id"]
            words = set(doc.text.lower().split())
            self.recipe_wordsets[recipe_id] = words
        self.is_trained = True

    def recommend_top1(self, query, valid_recipes_df) -> pd.DataFrame:
        if not self.is_trained:
            raise ValueError("JaccardRecommender not trained. Call train(...) first.")

        query_words = set(query.lower().split())

        best_row = None
        best_score = 0.0

        for _, row in valid_recipes_df.iterrows():
            recipe_id = row["recipe_id"]
            recipe_words = self.recipe_wordsets[recipe_id]
            inter = len(query_words.intersection(recipe_words))
            union = len(query_words.union(recipe_words))
            score = inter / union if union else 0
            if score > best_score:
                best_score = score
                best_row = row

        return best_row.to_frame()


class BigramOverlapRecommender(BaseRecommender):
    """
    1) Pre-store each doc's bigrams: {recipe_id -> set_of_bigrams}.
    2) For recommendation on valid_recipes_df:
       - Build query bigrams once, pick top overlap.
    """

    @staticmethod
    def _get_bigrams(text):
        words = text.lower().split()
        return set(tuple(words[i: i + 2]) for i in range(len(words) - 1))

    def __init__(self):
        self.recipe_bigrams = {}  # {recipe_id -> set_of_bigrams}
        self.is_trained = False

    def train(self, recipe_docs):
        for doc in recipe_docs:
            recipe_id = doc.metadata["recipe_id"]
            self.recipe_bigrams[recipe_id] = self._get_bigrams(doc.text)
        self.is_trained = True

    def recommend_top1(self, query, valid_recipes_df) -> pd.DataFrame:
        if not self.is_trained:
            raise ValueError("BigramOverlapRecommender not trained. Call train(...) first.")

        query_bigrams = self._get_bigrams(query)

        best_row = None
        best_score = -1

        for _, row in valid_recipes_df.iterrows():
            recipe_id = row["recipe_id"]
            overlap = len(query_bigrams.intersection(self.recipe_bigrams[recipe_id]))
            if overlap > best_score:
                best_score = overlap
                best_row = row

        return best_row.to_frame()


class BoWCosineRecommender(BaseRecommender):
    """
    1) Build a vocabulary from recipe_docs. Then store each doc's BOW vector: {recipe_id -> array}.
    2) For recommendation on valid_recipes_df:
       - Build query vector once, pick top cosine.
    """

    def __init__(self):
        self.vocab = []
        self.recipe_vectors = {}  # {recipe_id -> list/array}
        self.is_trained = False

    def train(self, recipe_docs):
        all_texts = [doc.text.lower().split() for doc in recipe_docs]
        # Build a sorted unique vocabulary
        self.vocab = sorted(set(word for text in all_texts for word in text))

        # Precompute BOW for each doc
        for doc in recipe_docs:
            recipe_id = doc.metadata["recipe_id"]
            self.recipe_vectors[recipe_id] = self._bow_vector(doc.text)

        self.is_trained = True

    def _bow_vector(self, text):
        words = text.lower().split()
        counts = Counter(words)
        return [counts.get(w, 0) for w in self.vocab]

    @staticmethod
    def _cosine(vec1, vec2):
        dot = sum(a * b for a, b in zip(vec1, vec2))
        mag1 = math.sqrt(sum(a * a for a in vec1))
        mag2 = math.sqrt(sum(b * b for b in vec2))
        return dot / (mag1 * mag2) if mag1 and mag2 else 0

    def recommend_top1(self, query, valid_recipes_df) -> pd.DataFrame:
        if not self.is_trained:
            raise ValueError("BoWCosineRecommender not trained. Call train(...) first.")

        query_vec = self._bow_vector(query)

        best_row = None
        best_score = -1

        for _, row in valid_recipes_df.iterrows():
            recipe_id = row["recipe_id"]
            recipe_vec = self.recipe_vectors[recipe_id]
            score = self._cosine(query_vec, recipe_vec)
            if score > best_score:
                best_score = score
                best_row = row

        return best_row.to_frame()


class Doc2VecRecommender(BaseRecommender):
    """
    1) Train a Doc2Vec model on recipe_docs. We store each doc's vector: {recipe_id -> np.array}.
    2) For recommendation on valid_recipes_df:
       - Infer query vector, pick top cosine from the subset.
    """

    def __init__(self, vector_size=50, window=2, min_count=1, epochs=20):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.epochs = epochs

        self.model = None
        self.recipe_vectors = {}  # {recipe_id -> np.array}
        self.is_trained = False

    def train(self, recipe_docs):
        documents = []
        for doc in recipe_docs:
            recipe_id = doc.metadata["recipe_id"]
            tokens = doc.text.lower().split()
            documents.append(TaggedDocument(tokens, [recipe_id]))

        # Train Doc2Vec
        self.model = Doc2Vec(
            documents=documents,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            epochs=self.epochs
        )

        # Precompute each doc vector
        for doc in recipe_docs:
            recipe_id = doc.metadata["recipe_id"]
            tokens = doc.text.lower().split()
            self.recipe_vectors[recipe_id] = self.model.infer_vector(tokens)

        self.is_trained = True

    def recommend_top1(self, query, valid_recipes_df) -> pd.DataFrame:
        if not self.is_trained:
            raise ValueError("Doc2VecRecommender not trained. Call train(...) first.")

        query_tokens = query.lower().split()
        query_vec = self.model.infer_vector(query_tokens)

        best_row = None
        best_score = float("-inf")

        for _, row in valid_recipes_df.iterrows():
            recipe_id = row["recipe_id"]
            recipe_vec = self.recipe_vectors[recipe_id]
            num = np.dot(query_vec, recipe_vec)
            denom = np.linalg.norm(query_vec) * np.linalg.norm(recipe_vec)
            sim = num / denom if denom else 0
            if sim > best_score:
                best_score = sim
                best_row = row

        return best_row.to_frame()


class PrecomputedEmbeddingRecommender(BaseRecommender):
    """
    If you already have embeddings for each doc, store them: {recipe_id -> np.array}.
    For recommendation on valid_recipes_df:
      - Use the given query embedding
      - Loop over subset, pick top cosine
    """

    def __init__(self):
        self.recipe_embeddings = {}  # {recipe_id -> np.array}
        self.is_trained = False

    def train(self, recipe_docs):
        """
        Suppose each doc has doc.embedding.
        We store doc.metadata["recipe_id"] -> doc.embedding
        """
        for doc in recipe_docs:
            recipe_id = doc.metadata["recipe_id"]
            self.recipe_embeddings[recipe_id] = doc.embedding
        self.is_trained = True

    def recommend_top1(self, query, valid_recipes_df) -> pd.DataFrame:
        if not self.is_trained:
            raise ValueError("PrecomputedEmbeddingRecommender not trained. Call train(...) first.")

        # Embed the query
        query_embedding = embedding_helper.embed_query(query)

        best_row = None
        best_score = float("-inf")

        for _, row in valid_recipes_df.iterrows():
            recipe_id = row["recipe_id"]
            recipe_vec = self.recipe_embeddings[recipe_id]
            denom = np.linalg.norm(query_embedding) * np.linalg.norm(recipe_vec)
            score = np.dot(query_embedding, recipe_vec) / denom if denom else 0
            if score > best_score:
                best_score = score
                best_row = row

        return best_row.to_frame()


def matches_to_response(matches):
    """
    Convert a list of matches to a response. Note: Only works for top-1 matches.

    :param matches: List of matched recipe_ids.
    :return: List of responses.
    """
    docs_list = [match.to_dict() for match in matches]
    response_texts = ['\n\n'.join([json.dumps(doc, indent=2) for doc in docs.values()]) for docs in docs_list]

    return [Response(response=response_text,
                     metadata={'recommendation': docs})
            for response_text, docs in zip(response_texts, docs_list)]




class BM25Recommender(BaseRecommender):
    """
    Okapi BM25 recommender backed by `rank-bm25`.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75, epsilon: float = 0.25):
        """
        :param float k1: TF saturation. Typical 1.2–2.0.
        :param float b: Length norm. 0 = none, 1 = full.
        :param float epsilon: Small constant for IDF floor.
        """
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon

        self.bm25 = None
        self.recipe_ids = []          # index -> recipe_id
        self.index_of = {}            # recipe_id -> index
        self.is_trained = False

    @staticmethod
    def _tokenize(text: str):
        """
        :param str text: Input text.
        :return: Lowercased whitespace tokens.
        :rtype: List[str]
        """
        return text.lower().split()

    def train(self, recipe_docs):
        """
        Fit BM25 over the recipe corpus.

        :param recipe_docs: Docs with ``doc.text`` and ``doc.metadata['recipe_id']``.
        """
        corpus_tokens = []
        self.recipe_ids = []

        for doc in recipe_docs:
            rid = doc.metadata["recipe_id"]
            tokens = self._tokenize(doc.text)
            corpus_tokens.append(tokens)
            self.recipe_ids.append(rid)

        self.index_of = {rid: i for i, rid in enumerate(self.recipe_ids)}
        self.bm25 = BM25Okapi(
            corpus_tokens, k1=self.k1, b=self.b, epsilon=self.epsilon
        )
        self.is_trained = True

    def recommend_top1(self, query, valid_recipes_df) -> pd.DataFrame:
        """
        Recommend the best match from a candidate subset.

        :param str query: Query string.
        :param pandas.DataFrame valid_recipes_df: Candidates. Must include ``recipe_id``.
        :return: Single-row DataFrame with the top recipe. Empty if no overlap.
        :rtype: pandas.DataFrame
        :raises ValueError: If not trained.
        """
        if not self.is_trained:
            raise ValueError("BM25Recommender not trained. Call train(...) first.")

        q_tokens = self._tokenize(query)
        # Scores for all docs once
        all_scores = self.bm25.get_scores(q_tokens)

        best_row = None
        best_score = float("-inf")

        for _, row in valid_recipes_df.iterrows():
            rid = row["recipe_id"]
            idx = self.index_of.get(rid)
            if idx is None:
                # Candidate not in index. Skip.
                continue
            score = all_scores[idx]
            if score > best_score:
                best_score = score
                best_row = row

        if best_row is None:
            # No candidate present in the index.
            return valid_recipes_df.iloc[0:0]

        return best_row.to_frame()
