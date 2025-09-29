import numpy as np
import pandas as pd
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.notebook import tqdm

from src.utils import hummus_util

# Define the weights for each feature
DEFAULT_FEATURES_WEIGHTS = {
    'title': 0.2,
    'description': 0.05,
    'directions': 0.05,
    'tags': 0.1,
    'ingredient_names': 0.6
}


def calc_tfidf_similarities_and_terms(pp_recipes, cluster_feature, tokenizer):
    cosine_similarities, tfidf_matrices, terms = _calc_tfidf_similarities(pp_recipes, cluster_feature=cluster_feature,
                                                                          min_df=0.0, max_df=0.8, max_features=200000,
                                                                          stop_words='english', use_idf=True,
                                                                          tokenizer=tokenizer, ngram_range=(1, 3))
    return cosine_similarities, tfidf_matrices, terms


def calc_tfidf_similarities(pp_recipes, feature_weights=None):
    cosine_similarities, tfidf_matrices, terms = _calc_tfidf_similarities(pp_recipes, feature_weights=feature_weights)
    return cosine_similarities


def _calc_tfidf_similarities(pp_recipes, feature_weights=None, min_df=0.0, max_df=1.0, max_features=None,
                             stop_words='english', use_idf=True, tokenizer=None, ngram_range=(1, 3),
                             cluster_feature=None):
    similarities = {}

    # Define the weights for each feature
    if feature_weights is None:
        feature_weights = DEFAULT_FEATURES_WEIGHTS

    # Create a TF-IDF vectorizers for the recipes
    tfidf_matrices, vectorizers = _vectorize_words(pp_recipes.copy(), feature_weights.keys(), min_df, max_df,
                                                   max_features, stop_words, use_idf, tokenizer, ngram_range)

    # Add all terms
    terms = {key: value.get_feature_names_out() for key, value in vectorizers.items()}

    # build vocabulary
    for feature, matrix in tfidf_matrices.items():
        if cluster_feature is None or cluster_feature == feature:
            similarities[feature] = cosine_similarity(tfidf_matrices[feature])

    # Initialize an zero matrix for the overall similarities
    length = list(tfidf_matrices.items())[0][1].shape[0]
    cosine_similarities = np.zeros((length, length))

    # Compute the overall similarity for each document
    for feature, matrix in tfidf_matrices.items():
        if cluster_feature is None or cluster_feature == feature:
            feature_weight = feature_weights[feature]
            feature_similarities = similarities[feature]
            weighted_feature_similarities = feature_weight * feature_similarities
            cosine_similarities = cosine_similarities + weighted_feature_similarities

    return cosine_similarities, tfidf_matrices, terms


def _vectorize_words(pp_recipes, features, min_df, max_df, max_features, stop_words, use_idf, tokenizer, ngram_range):
    tfidf_matrices = {}
    vectorizers = {}

    for feature in features:
        # if feature is not string but enumerable of strings, then append them to one string
        if isinstance(pp_recipes[feature][0], list):
            pp_recipes[feature] = pp_recipes[feature].apply(lambda x: ' '.join(s for s in x if isinstance(s, str)))

        tf = TfidfVectorizer(min_df=min_df, max_df=max_df, max_features=max_features, stop_words=stop_words,
                             use_idf=use_idf, tokenizer=tokenizer, ngram_range=ngram_range)

        # Replace Nan with empty string for current feature
        pp_recipes[feature].fillna('', inplace=True)

        tfidf_matrix = tf.fit_transform(pp_recipes[feature])
        tfidf_matrices[feature] = tfidf_matrix
        vectorizers[feature] = tf

    return tfidf_matrices, vectorizers


def get_healthy_substitutions(config_grid, cosine_similarities):
    # Start the timer
    start_time = time.time()
    healthy_substitution_pairs = []

    for config in tqdm(config_grid, desc="Substitution per configs"):
        # Unpack current model & post-processing parameters
        (score_name, recipe_to_score), substitution_threshold \
            = config['score_config'], config['substitution_threshold']

        # Define the normalization dicts for the scores
        # if score_name == score_util.NUTRI_SCORE: Right now nutri score is in [0,1], so this dict is not needed
        #     score_norm = {"A": 4, "B": 3, "C": 2, "D": 1, "E": 0}
        # else:
        score_norm = hummus_util.IdDict()

        # Get "best" (healthiest and best matching) cosine similarities for each recipe
        healthy_substitutions = {}
        for recipeID in tqdm(range(cosine_similarities.shape[0]), leave=False, desc="Similarities per recipe"):
            # Default case: Recipe is not substituted
            healthy_substitutions[recipeID] = recipeID

            # Get the most similar recipes
            recipe_similarities = pd.DataFrame(cosine_similarities[recipeID]).sort_values(by=0, ascending=False)
            recipe_similarities = recipe_similarities.drop(recipeID, axis=0)  # Remove the recipe itself

            # Search a healthier recipe while being in bounds of the substitution similarity threshold
            for substitution_recipe in recipe_similarities.index:
                if recipe_similarities.loc[substitution_recipe][0] >= substitution_threshold:
                    if score_norm[recipe_to_score[substitution_recipe]] > score_norm[recipe_to_score[recipeID]]:
                        healthy_substitutions[recipeID] = substitution_recipe
                        break
                else:  # as recipes are sorted descending, we can break as soon as the similarity is too low
                    break

        # Add result
        healthy_substitution_pairs.append((healthy_substitutions, config))

    # End the timer
    end_time = time.time()
    elapsed_time = end_time - start_time

    print("It took {} minutes for processing the healthy substitutions.".format(elapsed_time / 60))
    return healthy_substitution_pairs


def get_similar_recipes(recipes_df, cosine_similarities, similarity_thresholds):
    similar_recipes_dict = {}

    # Set default params
    if similarity_thresholds is None:
        similarity_thresholds = [0.1]

    for similarity_threshold in similarity_thresholds:
        by_id = {}
        by_name = {}
        by_url = {}

        for recipeID in range(cosine_similarities.shape[0]):
            # Get the most similar recipes
            recipe_similarities = pd.DataFrame(cosine_similarities[recipeID]).sort_values(by=0, ascending=False)
            recipe_similarities = recipe_similarities.drop(recipeID, axis=0)  # Remove the recipe itself
            similar_recipes = recipe_similarities[recipe_similarities[0] >= similarity_threshold].index.tolist()
            by_id[recipeID] = similar_recipes

            # dict by name
            recipe_name = recipes_df.loc[recipes_df['recipe_id'] == recipeID]['title'].iloc[0]
            by_name[recipe_name] = recipes_df.loc[recipes_df['recipe_id'].isin(similar_recipes), 'title'].tolist()

            # dict by url
            recipe_url = recipes_df.loc[recipes_df['recipe_id'] == recipeID]['recipe_url'].iloc[0]
            by_url[recipe_url] = recipes_df.loc[
                recipes_df['recipe_id'].isin(similar_recipes), 'recipe_url'].tolist()

        # Add the results to the dictionary
        similar_recipes_dict[similarity_threshold] = {
            'by_id': by_id,
            'by_name': by_name,
            'by_url': by_url
        }

    return similar_recipes_dict
