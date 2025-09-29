import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD


def full_clustering(recipes, reviews, users, n_clusters=10, random_state=42):
    # Assuming you have hummus_data as input
    user_clusters, user_features_sparse = _cluster(reviews, n_clusters=n_clusters, random_state=random_state)

    # Create cluster-users DataFrame
    user_cluster_dict = _create_clustered_users(reviews, user_clusters)

    # Update the data
    recipes, reviews, users = _update_data(recipes, reviews, users, user_cluster_dict)

    return recipes, reviews, users, user_cluster_dict, user_clusters, user_features_sparse


# N_clusters is the maximum number of clusters to be created, some clusters may be empty and therefore not created
def _cluster(review_df, n_clusters=10, random_state=42):
    all_interactions = review_df[['member_id', 'recipe_id', 'rating']]
    user_features_sparse = to_sparse_interactions(all_interactions)

    # Train the model and use model to cluster users
    user_clusters = MiniBatchKMeans(n_clusters=n_clusters, batch_size=3072, random_state=random_state).fit_predict(
        user_features_sparse)

    return user_clusters, user_features_sparse


def to_sparse_interactions(all_interactions, remove_invalid_interactions=True, clean_negative_ratings=True):
    if remove_invalid_interactions:  # Remove all invalid interactions (any negative feature values)
        all_interactions = all_interactions[
            (all_interactions['member_id'] >= 0) & (all_interactions['recipe_id'] >= 0)]

    if clean_negative_ratings:  # Set all negative ratings to 0
        all_interactions.loc[all_interactions['rating'] < 0, 'rating'] = 0

    # Convert float64 datatype of the rating column to float32 to improve performance
    all_interactions['rating'] = all_interactions['rating'].astype(np.float32)

    # Convert the DataFrame to a sparse matrix directly
    user_features_sparse = csr_matrix(
        (all_interactions['rating'], (all_interactions['member_id'], all_interactions['recipe_id'])))

    return user_features_sparse


def to_sparse_user_interactions(user_interactions, num_recipes, remove_invalid_interactions=True,
                                clean_negative_ratings=True):
    if remove_invalid_interactions:  # Remove all invalid interactions (any negative feature values)
        user_interactions = user_interactions[user_interactions['recipe_id'] >= 0]

    if clean_negative_ratings:
        # Set all negative ratings to 0
        user_interactions.loc[user_interactions['rating'] < 0, 'rating'] = 0

    # Convert float64 datatype of the rating column to float32 to improve performance
    user_interactions['rating'] = user_interactions['rating'].astype(np.float32)

    # Convert the DataFrame to a sparse matrix directly
    user_features_sparse = csr_matrix(
        (user_interactions['rating'], (np.zeros(len(user_interactions)), user_interactions['recipe_id'])),
        shape=(1, num_recipes))

    return user_features_sparse


def _create_clustered_users(reviews, user_clusters, reset_index=True):
    # Initialize the dictionary to map the old member_id to the new cluster_id
    user_cluster_dict = {}

    # Extract interactions DataFrame
    all_interactions = reviews[['member_id', 'recipe_id', 'rating']]

    for cluster_id in np.unique(user_clusters):
        # Identify users in the current cluster
        cluster_members = np.where(user_clusters == cluster_id)[0]

        # Create a new user ID for the cluster
        if reset_index:
            new_cluster_user_id = cluster_id
        else:
            new_cluster_user_id = all_interactions['member_id'].max() + 1 + cluster_id

        # Update the dict
        for old_member_id in cluster_members:
            assert old_member_id not in user_cluster_dict  # Each user can be only in one cluster
            user_cluster_dict[old_member_id] = new_cluster_user_id

    return user_cluster_dict


def _update_data(recipes, reviews, users, user_cluster_dict):
    # Copy the original dataframes
    reviews = reviews.copy()
    recipes = recipes.copy()
    users = users.copy()

    # Update the recipie authors with the new cluster-users
    _update_user_ids(recipes, 'author_id', user_cluster_dict)
    recipes = recipes.reset_index(drop=True)

    # Update the reviews with the new cluster-users.
    # Merge reviews for the same cluster and recipe by calculating an average rating.
    _update_user_ids(reviews, 'member_id', user_cluster_dict)
    agg_reviews = {
        'rating': 'mean',
        'text': 'first',
        'likes': 'sum',
        'last_modified_date': 'max'
    }
    reviews = reviews.groupby(['member_id', 'recipe_id'], as_index=False).agg(agg_reviews)
    reviews = reviews.reset_index(drop=True)

    # Update the members with the new cluster-users. Keep the first member-entity for each cluster.
    _update_user_ids(users, 'member_id', user_cluster_dict)
    agg_users = {
        'member_url': 'first',
        'member_name': 'first',
        'member_description': 'first',
        'member_status': 'first',
        'member_avg_rating': 'mean',
        'member_map_pin': 'first',
        'member_joined': 'max',
        'follows_count': 'sum',
        'follow_me_count': 'sum',
        'avatar_url': 'first'
    }
    users = users.groupby('member_id', as_index=False).agg(agg_users)
    users = users.reset_index(drop=True)

    return recipes, reviews, users


def _update_user_ids(df, id_column, id_mapping):
    # Updates user IDs in a dataframe
    df[id_column] = df[id_column].map(id_mapping.get)


def visualize_clusters(user_clusters, user_features):
    try:
        # Visualize the clusters (for 2D data, reduce dimensionality using PCA)
        pca = PCA(n_components=2)
        user_features_pca = pca.fit_transform(user_features)
    except TypeError:
        # Sparse matrix as input won't work on PCA, take TruncatedSVD instead
        user_features_pca = TruncatedSVD(n_components=2).fit_transform(user_features)

    plt.scatter(user_features_pca[:, 0], user_features_pca[:, 1], c=user_clusters, cmap='viridis')
    plt.title('User Clusters Based on Item Interactions')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.show()
