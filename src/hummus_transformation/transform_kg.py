import ast
import multiprocessing
import numpy as np
import pandas as pd
import rdflib
from abc import ABC
from itertools import chain, islice
from pathlib import Path
from rdflib import Graph, URIRef, Literal, BNode
from typing import List

from src.food_rag.utils import hummus_docify
from src.food_rag.utils import neo4J_helper
from src.hummus_transformation import usda_fields
from src.utils import hummus_util
from src.utils import load_util
from src.utils import rdf_util

SUB_DIRECTORY_NAME = 'neo4j_kg'
CONFIG_FILE_NAME = 'neo4j_load_config.pkl'

# Define all modes
MODE_ALL = 'all'  # Includes usda data, food_on data, and hummus data, all linked together by the FoodKG links
MODE_NO_USDA = 'nousda'  # Includes food_on data and hummus data, all linked together by the FoodKG links
MODE_UNLINKED_HUMMUS = 'unlinkedhummus'  # Includes food_on data and hummus data, but not linked together
MODE_ONLY_FOOD_ON = 'onlyfoodon'  # Includes only the food_on data

# Define the URI prefix
FOOD_ON_PREFIX = rdf_util.FOOD_ON_PREFIX

# Define the specific objects or subjects URIRef
OWL_CLASS = URIRef(rdf_util.OWL_CLASS)
OWL_OBJECT_PROPERTY = URIRef(rdf_util.OWL_OBJECT_PROPERTY)
OWL_SYMMETRIC_PROPERTY = URIRef(rdf_util.OWL_SYMMETRIC_PROPERTY)

# Define the specific predicate URIRef
SUB_PROPERTY_OF_PREDICATE = URIRef(rdf_util.RDFS_SUB_PROPERTY)
TYPE_PREDICATE = URIRef(rdf_util.RDF_TYPE)
SUB_CLASS_OF_PREDICATE = URIRef(rdf_util.RDFS_SUBCLASS)
LABEL_PREDICATE = URIRef(rdf_util.RDFS_LABEL)
COMMENT_PREDICATE = URIRef(rdf_util.RDFS_COMMENT)
BROAD_SYNONYM_PREDICATE = URIRef("http://www.geneontology.org/formats/oboInOwl#hasBroadSynonym")
NARROW_SYNONYM_PREDICATE = URIRef("http://www.geneontology.org/formats/oboInOwl#hasNarrowSynonym")
EXACT_SYNONYM_PREDICATE = URIRef("http://www.geneontology.org/formats/oboInOwl#hasExactSynonym")
SYNONYM_PREDICATE = URIRef("http://www.geneontology.org/formats/oboInOwl#hasSynonym")
RELATED_SYNONYM_PREDICATE = URIRef("http://www.geneontology.org/formats/oboInOwl#hasRelatedSynonym")
OTHER_PREDICATES = {
    URIRef(rdf_util.OWL_DISJOINT_WITH),
    URIRef(rdf_util.RDFS_RANGE),
}

# Define the set of all relevant predicates
RELEVANT_PREDICATES = {SUB_PROPERTY_OF_PREDICATE} | {TYPE_PREDICATE} | {LABEL_PREDICATE} | {COMMENT_PREDICATE} | {
    SUB_CLASS_OF_PREDICATE} | {BROAD_SYNONYM_PREDICATE} | {NARROW_SYNONYM_PREDICATE} | {EXACT_SYNONYM_PREDICATE} | {
                          SYNONYM_PREDICATE} | {RELATED_SYNONYM_PREDICATE} | OTHER_PREDICATES

# Define the CSV delimiter
CSV_DELIMITER = '|'


class Neo4JTransformer:
    def __init__(self, mode=None):
        """
        Transform the Hummus data to fit the Neo4J graph database.

        :param mode: The mode can be either 'all', 'no_usda', or 'only_food_on'. If 'all', all data will be loaded. If
        'no_usda', all data will be loaded except the usda data. If 'only_food_on', only the food_on data will be
         loaded.
        """
        self.file_names_dict = {'mode': mode, 'files': {}}
        self.objects: List[Neo4JObject] = []

    def get_mode(self):
        return self.file_names_dict['mode']

    def save(self, data_path):
        load_util.ensure_dir(Path(data_path, SUB_DIRECTORY_NAME, self.get_mode()))
        load_util.store_file(self.file_names_dict,
                             Path(data_path, SUB_DIRECTORY_NAME, self.get_mode(), CONFIG_FILE_NAME))

        for obj in self.objects:
            obj.save(data_path, self.get_mode())

    @staticmethod
    def load(data_path, mode):
        # Load the file names dict
        file_names_dict = load_util.load_file(Path(data_path, SUB_DIRECTORY_NAME, mode, CONFIG_FILE_NAME))

        # Create a new Neo4JTransformer object
        neo4j_transformer = Neo4JTransformer(mode=mode)

        # Set file names dict
        neo4j_transformer.file_names_dict = file_names_dict

        # Set objects
        for name, object_type in file_names_dict['files'].items():
            neo4j_transformer.objects.append(Neo4JObject.load(data_path, mode, object_type, name))

        return neo4j_transformer

    def get_all_node_file_dicts(self):
        return {obj.get_file_name(): obj.name for obj in self.objects if obj.object_type == 'node'}

    def get_all_relation_file_dicts(self):
        return {obj.get_file_name(): obj.name for obj in self.objects if obj.object_type == 'relation'}

    def get_all_node_features(self):
        return {obj.name: list(obj.df.columns) for obj in self.objects if obj.object_type == 'node'}

    def get_all_cleaned_node_features(self):
        return {obj.name: [neo4J_helper.Neo4jManager.clean_column_name(feature) for feature in list(obj.df.columns)]
                for obj in self.objects if obj.object_type == 'node'}

    def get_all_relation_features(self):
        return {obj.name: list(obj.df.columns) for obj in self.objects if obj.object_type == 'relation'}

    def get_all_node_types(self):
        return [obj.name for obj in self.objects if obj.object_type == 'node']

    def get_all_relation_types(self):
        return [obj.name for obj in self.objects if obj.object_type == 'relation']

    def get_all_objects(self):
        return {obj.get_file_name(): obj.df for obj in self.objects}

    def get_all_nodes(self):
        return [obj.df for obj in self.objects if obj.object_type == 'node']

    def get_all_relations(self):
        return [obj.df for obj in self.objects if obj.object_type == 'relation']

    def _add_object(self, obj):
        self.objects.append(obj)
        self.file_names_dict['files'][obj.name] = obj.object_type

    def get_db_name(self):
        mode = self.get_mode()

        if mode is None or mode == MODE_ALL:
            return 'RecipesSpace_All'
        elif mode == MODE_NO_USDA:
            return 'RecipesSpace_NoUSDA'
        elif mode == MODE_UNLINKED_HUMMUS:
            return 'RecipesSpace_UnlinkedHummus'
        elif mode == MODE_ONLY_FOOD_ON:
            return 'RecipesSpace_OnlyFoodOn'
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def get_schema(self):
        schema = {
            'node_types': self.get_all_node_types(),
            'node_features': self.get_all_cleaned_node_features(),
            'relation_types': self.get_all_relation_types(),
            'relation_features': self.get_all_relation_features()
        }
        return schema

    def transform(self, hummus_data, limit_size=None, verbose=True, **kwargs):
        # Transform all data to fit the Neo4J
        if verbose:
            print(f"Transforming data to fit the Neo4J with mode: {self.get_mode()}.")

        # Set defaults
        if self.get_mode() is None:  # If mode is None, set mode to default 'all'
            self.file_names_dict['mode'] = MODE_ALL

        # If limit size is set to a number, then decreases the size of the data for debug purposes
        _limit_sizes(hummus_data, limit_size)

        # Transform the recipe, review, user and tag data
        if self.get_mode() != MODE_ONLY_FOOD_ON:
            self._recipes_to_kg(hummus_data['recipes_df'], verbose=verbose)
            self._members_to_kg(hummus_data['users_df'], hummus_data['recipes_df'], verbose=verbose)
            self._reviews_to_kg(hummus_data['reviews_df'], hummus_data['recipes_df'], hummus_data['users_df'],
                                verbose=verbose)
            self._tags_to_kg(hummus_data['recipes_df'], verbose=verbose)

        # Transform the food_on data
        node_to_id = self._food_on_to_kg(hummus_data['food_on_ontology'], verbose=verbose)

        # Transform the ingredient data which can be connected to either, none, or both food_on and usda data
        if self.get_mode() != MODE_ONLY_FOOD_ON:
            # Must be last since it needs to connect to the food_on kg
            self._ingredients_to_kg(hummus_data, node_to_id, verbose=verbose, **kwargs)

    def _recipes_to_kg(self, recipes_df, verbose=True):  # Shall not be executed in 'only_food_on' mode
        if verbose:
            print('Transform Recipe.')

        # Clean df
        recipes_df, _, _ = hummus_docify.clean_df(
            df=recipes_df,
            id_column="recipe_id",
            rename_columns={'duration': 'duration [min]', 'ingredients': 'ingredient_quantities',
                            'ingredient_names': 'ingredients', 'direction_size': 'number_of_directions',
                            'ingredients_sizes': 'number_of_ingredients'},
            drop_columns=['normalization_comment'],
            string_columns=['title', 'description', 'recipe_url', 'servingsPerRecipe', 'serves'],
            number_columns=['duration [min]', 'average_rating', 'number_of_ratings', 'servingSize [g]',
                            'calories [kcal]',
                            'caloriesFromFat [kcal]', 'totalFat [g]', 'saturatedFat [g]', 'cholesterol [mg]',
                            'sodium [mg]',
                            'totalCarbohydrate [g]', 'dietaryFiber [g]', 'sugars [g]', 'protein [g]',
                            'number_of_directions', 'number_of_ingredients', 'who_score', 'fsa_score', 'nutri_score'],
            list_columns=['directions', 'tags', 'ingredients'],
            date_columns=['last_changed_date'],
            included_content_features=['type', 'title', 'description', 'serves', 'directions', 'ingredients',
                                       'ingredient_quantities', 'tags', 'average_rating', 'number_of_ratings',
                                       'servingSize [g]', 'calories [kcal]', 'caloriesFromFat [kcal]', 'totalFat [g]',
                                       'saturatedFat [g]', 'cholesterol [mg]', 'sodium [mg]', 'totalCarbohydrate [g]',
                                       'dietaryFiber [g]', 'sugars [g]', 'protein [g]', 'who_score', 'fsa_score',
                                       'nutri_score', 'recipe_url'],
            content_functions={
                'directions': lambda x: '\n'.join([f"{step + 1}. {instruction}" for step, instruction in enumerate(x)]),
                'ingredient_quantities': hummus_docify.stringify_nested_dict,
            },
            string_default='not specified',
            number_default=-1,
            stringify_columns=['directions', 'tags', 'ingredients', 'last_changed_date']
        )

        # Replace all double quotes in the DataFrame
        recipes_df = recipes_df.replace('"', '', regex=True)

        # Replace all "|" in the DataFrame
        recipes_df = recipes_df.replace('|', '', regex=True)

        # Convert recipes_df to nodes
        recipes_nodes = transform_to_node_df(df=recipes_df)

        # Add to graph
        self._add_object(Neo4JNode(node_type='Recipe', id_column='recipe_id', df=recipes_nodes))

    def _members_to_kg(self, members_df, recipes_df, verbose=True):  # Shall not be executed in 'only_food_on' mode
        if verbose:
            print('Transform Member.')

        # Clean df
        members_df, _, _ = hummus_docify.clean_df(
            members_df,
            id_column="user_id",
            rename_columns={'member_id': 'user_id', 'member_url': 'url', 'member_name': 'name',
                            'member_description': 'description', 'member_status': 'status',
                            'member_avg_rating': 'avg_rating', 'member_map_pin': 'map_pin',
                            'member_joined': 'joined_at'},
            drop_columns=[],
            string_columns=['url', 'name', 'description', 'status', 'map_pin', 'avatar_url'],
            number_columns=['avg_rating', 'follows_count', 'follow_me_count'],
            list_columns=[],
            date_columns=['joined_at'],
            included_content_features=['type', 'name', 'description', 'status', 'map_pin'],
            content_functions={},
            string_default='not specified',
            number_default=-1,
            stringify_columns=['joined_at']
        )

        # Convert members_df to nodes
        members_nodes = transform_to_node_df(df=members_df)

        # remove author ids which are nan and -1 as they do not exist in the current recipe data (only influences the
        # recipe_member_relationships)
        recipes_df = recipes_df[recipes_df['author_id'] != -1].dropna(subset=['author_id'])

        # Member-Recipe relationships (Authorship)
        recipe_member_relationships = transform_to_relation_df(recipes_df, 'recipe_id', 'author_id')

        # Add to graph
        self._add_object(Neo4JNode(node_type='Member', id_column='user_id', df=members_nodes))
        self._add_object(Neo4JRelation(relation_type='HAS_AUTHOR',
                                       src_id_column='recipe_id', src_node_type='Recipe',
                                       dst_id_column='author_id', dst_node_type='Member', dst_id_name='user_id',
                                       df=recipe_member_relationships))

    def _reviews_to_kg(self, reviews_df, recipes_df, members_df, verbose=True):
        # Shall not be executed in 'only_food_on' mode

        if verbose:
            print('Transform Review.')

        # Clean df
        reviews_df, _, _ = hummus_docify.clean_df(
            reviews_df,
            id_column=None,  # We just take the index as the ID, is automatically added and referenced as 'id'
            rename_columns={'id': 'review_id', 'member_id': 'user_id'},
            string_columns=['text'],
            number_columns=['rating', 'likes'],
            list_columns=[],
            date_columns=['last_modified_date'],
            included_content_features=['review_id', 'recipe_id', 'user_id', 'rating', 'text', 'likes'],
            content_functions={},
            string_default='not specified',
            number_default=-1,
            stringify_columns=['last_modified_date']
        )

        # Add a name column
        name_series = (reviews_df.merge(recipes_df[['recipe_id', 'title']], on='recipe_id', how='left')
                       .merge(members_df[['member_id', 'member_name']], left_on='user_id', right_on='member_id',
                              how='left')
                       .apply(lambda row: f'Review {f"for {row.title} " if not pd.isna(row.title) else ""}'
                                          f'{f"by {row.member_name} " if not pd.isna(row.member_name) else ""}'
                                          f'({row.review_id})', axis=1))
        reviews_df['name'] = name_series

        # Convert reviews_df to nodes
        reviews_nodes = transform_to_node_df(df=reviews_df)

        # Member-Review relationships
        review_member_relationships = transform_to_relation_df(reviews_df, 'review_id', 'user_id')

        # Review-Recipe relationships
        recipe_review_relationships = transform_to_relation_df(reviews_df, 'recipe_id', 'review_id')

        # Add to graph
        self._add_object(Neo4JNode(node_type='Review', id_column='review_id', df=reviews_nodes))
        self._add_object(Neo4JRelation(relation_type='WRITTEN_BY',
                                       src_id_column='review_id', src_node_type='Review',
                                       dst_id_column='user_id', dst_node_type='Member',
                                       df=review_member_relationships))
        self._add_object(Neo4JRelation(relation_type='HAS_REVIEW',
                                       src_id_column='recipe_id', src_node_type='Recipe',
                                       dst_id_column='review_id', dst_node_type='Review',
                                       df=recipe_review_relationships))

    def _tags_to_kg(self, recipes_df, verbose=True):  # Shall not be executed in 'only_food_on' mode
        if verbose:
            print('Transform Tag.')

        # Extract unique tags
        expanded_df = recipes_df.explode('tags')
        expanded_df['tags'] = expanded_df['tags'].fillna('empty-tag')
        expanded_df = expanded_df[expanded_df['tags'] != '']
        tags_df = expanded_df[['tags', 'recipe_id', 'author_id']].drop_duplicates()
        tags_df = tags_df.reset_index(drop=True)  # reset index

        # for each unique tag assign a unique id
        tags_df['tag_id'] = tags_df['tags'].map({tag: i for i, tag in enumerate(tags_df['tags'].unique(), 0)})

        # Rename tags column to name
        tags_df.rename(columns={'tags': 'name'}, inplace=True)

        # Convert tags_df to nodes
        tags_nodes = transform_to_node_df(df=tags_df.drop(columns=['recipe_id', 'author_id'])
                                          .drop_duplicates().reset_index(drop=True))

        # Recipe-Tag relationships
        recipe_tag_relationships = transform_to_relation_df(tags_df, 'recipe_id', 'tag_id')

        # Add to graph
        self._add_object(Neo4JNode(node_type='Tag', id_column='tag_id', df=tags_nodes))
        self._add_object(Neo4JRelation(relation_type='HAS_TAG',
                                       src_id_column='recipe_id', src_node_type='Recipe',
                                       dst_id_column='tag_id', dst_node_type='Tag',
                                       df=recipe_tag_relationships))

    def _food_on_to_kg(self, graph, verbose=True):
        # Applicable for all modes ('all', 'no_usda', 'unlinked_hummus', 'only_food_on')
        if verbose:
            print('Transform Food_on.')

        # Preprocess graph
        pp_graph, node_types_dict, relationship_types_dict = _preprocess_food_on(graph)

        # Convert to DataFrame
        node_to_id, rdf_nodes_df, rdf_relationships_df = _convert_to_df(pp_graph, node_types_dict,
                                                                        relationship_types_dict)

        # Cleaning
        default_synonym = 'not specified'
        for col in ['label', 'comment', 'broad_synonym', 'narrow_synonym', 'synonym', 'exact_synonym',
                    'related_synonym']:
            rdf_nodes_df[col].replace({'': default_synonym, None: default_synonym}, inplace=True)

        # Convert rdf_nodes_df to nodes
        rdf_nodes = transform_to_node_df(df=rdf_nodes_df)

        # rdf relationships
        rdf_node_relationships_df = transform_to_relation_df(rdf_relationships_df, 'src_id', 'dst_id')

        # Add to graph
        self._add_object(Neo4JNode(node_type='Food_on', id_column='id', df=rdf_nodes))
        self._add_object(Neo4JRelation(relation_type='SUB_CLASS_OF',
                                       src_id_column='src_id', src_node_type='Food_on', src_id_name='id',
                                       dst_id_column='dst_id', dst_node_type='Food_on', dst_id_name='id',
                                       df=rdf_node_relationships_df))

        return node_to_id

    def _ingredients_to_kg(self, hummus_data, node_to_id, verbose=True, **kwargs):
        # Shall not be executed in 'only_food_on' mode
        if verbose:
            print('Transform Ingredient.')

        # Get nutrient information for the ingredients from the usda dataset
        if self.get_mode() == MODE_ALL:
            id_columns, text_columns, nutrient_columns, usda_df = _get_usda(hummus_data, **kwargs)
        else:  # mode == 'no_usda'
            id_columns, text_columns, nutrient_columns = [], [], []
            usda_df = None

        # Get ingredients and set their type
        ingredients_df = hummus_data['ingredients_df']

        # Map ingredient_id to ingredient_url using ingredients_dict
        ingredients_df['ingredient_url'] = ingredients_df['ingredient_id'].map(hummus_data['ingredients_dict'])

        # Map ingredient_url to usda_fdcId using usda_mappings
        usda_mappings = {k: int(v.split('#')[-1]) for k, v in hummus_data['usda_mappings'].items()}
        ingredients_df['ndb_number'] = ingredients_df['ingredient_url'].map(usda_mappings)

        # Recipe-Ingredient relationships
        recipe_ingredient_relationships = ingredients_df[['recipe_id', 'ingredient_id']]
        recipe_ingredient_relationships['relationship_type'] = 'has_ingredient'
        recipe_ingredient_relationships.columns = ['recipe_id', 'ingredient_id', 'relationship_type']  # rename columns

        # After processing the relationships remove the recipe_id column and remove duplicates
        ingredients_df = ingredients_df.drop(columns=['recipe_id']).drop_duplicates()

        # Merge nutrient information from usda_df to ingredients_df
        if self.get_mode() == MODE_ALL:
            ingredients_df = pd.merge(ingredients_df, usda_df, left_on='ndb_number', right_on='ndb_number', how='left')

        # Clean df
        ingredients_df, _, _ = hummus_docify.clean_df(
            ingredients_df,
            id_column='ingredient_id',
            # We just take the index as the ID, is automatically added and referenced as 'id'
            rename_columns={'ing_name': 'name'},
            string_columns=['name'] + text_columns,
            number_columns=id_columns + nutrient_columns,
            list_columns=[],
            date_columns=[],
            included_content_features=['name'] + id_columns + text_columns + nutrient_columns,
            string_default='not specified',
            number_default=-1,
            stringify_columns=[]
        )

        # Convert ingredients_nodes
        ingredients_nodes = transform_to_node_df(df=ingredients_df)

        # Recipe-Ingredient relationships
        recipe_ingredient_node_relationships = transform_to_relation_df(recipe_ingredient_relationships, 'recipe_id',
                                                                        'ingredient_id')

        # Add to graph
        self._add_object(Neo4JNode(node_type='Ingredient', id_column='ingredient_id', df=ingredients_nodes))
        self._add_object(Neo4JRelation(relation_type='HAS_INGREDIENT',
                                       src_id_column='recipe_id', src_node_type='Recipe',
                                       dst_id_column='ingredient_id', dst_node_type='Ingredient',
                                       df=recipe_ingredient_node_relationships))

        if self.get_mode() != MODE_UNLINKED_HUMMUS:
            # Connect the ingredients kg with the foodOn kg
            ingredient_food_on_relationships = _connect_kgs(ingredients_df, hummus_data['ingredients_dict'],
                                                            hummus_data['food_on_mappings'], node_to_id)

            # Ingredient-FoodOn relationships
            ingredient_food_on_node_relationships = transform_to_relation_df(ingredient_food_on_relationships,
                                                                             'ingredient_id', 'food_on_id')

            # Add to graph
            self._add_object(Neo4JRelation(relation_type='IS_EQUIVALENT_TO',
                                           src_id_column='ingredient_id', src_node_type='Ingredient',
                                           dst_id_column='food_on_id', dst_node_type='Food_on', dst_id_name='id',
                                           df=ingredient_food_on_node_relationships))


class Neo4JObject(ABC):
    def __init__(self, name, object_type, df=None, metadata=None):
        self.name = name
        self.object_type = object_type
        self.df = df
        self.metadata = metadata if metadata is not None else {}

        if df is not None:
            self.metadata.update({
                'headers': list(df.columns),
                'dtypes': df.dtypes.to_dict(),
                'object_type': self.object_type,
                'name': self.name
            })

    def save(self, path, mode):
        if self.df is not None:
            load_util.ensure_dir(f'{path}/{SUB_DIRECTORY_NAME}/{mode}')
            self.df.to_csv(get_file_path(path, mode, self.object_type, self.name), index=False, sep=CSV_DELIMITER,
                           na_rep='NULL')
            load_util.store_file(self.metadata, get_file_metadata_path(path, mode, self.object_type, self.name))

    @staticmethod
    def load(path, mode, object_type, name):
        df = pd.read_csv(get_file_path(path, mode, object_type, name), sep=CSV_DELIMITER, keep_default_na=False)
        df = df.replace('NULL', None)  # Replace all NULL values with None

        metadata = load_util.load_file(get_file_metadata_path(path, mode, object_type, name))

        if object_type == 'node':
            neo4j_object = Neo4JNode(node_type=metadata['node_type'], id_column=metadata['id_column'], df=df,
                                     metadata=metadata)
        elif object_type == 'relation':
            neo4j_object = Neo4JRelation(relation_type=metadata['relation_type'],
                                         src_id_column=metadata['src_id_column'],
                                         src_node_type=metadata['src_node_type'],
                                         dst_id_column=metadata['dst_id_column'],
                                         dst_node_type=metadata['dst_node_type'],
                                         df=df, metadata=metadata)
        else:
            raise ValueError(f"Unknown object type: {object_type}")

        return neo4j_object

    def get_file_name(self):
        return f'{self.object_type}_{self.name}.csv'


class Neo4JNode(Neo4JObject):
    def __init__(self, node_type: str, id_column: str, df=None, metadata=None):
        super().__init__(name=node_type, object_type='node', df=df, metadata=metadata)
        self.metadata.update({'node_type': node_type, 'id_column': id_column})


class Neo4JRelation(Neo4JObject):
    def __init__(self, relation_type: str, src_id_column: str, src_node_type: str, dst_id_column: str,
                 dst_node_type: str, src_id_name: str = None, dst_id_name: str = None, df=None, metadata=None):
        """
        Create a Neo4JRelation object.

        :param relation_type: The type of the relation.
        :param src_id_column: The column name of the source id in the DataFrame.
        :param src_node_type: The node type of the source node.
        :param dst_id_column: The column name of the destination id in the DataFrame.
        :param dst_node_type: The node type of the destination node.
        :param src_id_name: The column name of the src id in the respective node DataFrame. If None, it will
        be set to the same as src_id_column.
        :param dst_id_name: The column name of the dst id in the respective node DataFrame. If None, it will
        be set to the same as dst_id_column.
        :param df: The DataFrame containing the relation data. Optional and can be set later.
        :param metadata: The metadata of the relation. Optional and will be extended with the relation specific
        metadata.
        """
        super().__init__(name=relation_type, object_type='relation', df=df, metadata=metadata)

        if src_id_name is None:
            src_id_name = src_id_column
        if dst_id_name is None:
            dst_id_name = dst_id_column

        self.metadata.update({
            'relation_type': relation_type,
            'src_id_column': src_id_column,
            'src_id_name': src_id_name,
            'src_node_type': src_node_type,
            'dst_id_column': dst_id_column,
            'dst_node_type': dst_node_type,
            'dst_id_name': dst_id_name
        })


def get_file_path(data_path, mode, type_name, name):
    return Path(data_path, SUB_DIRECTORY_NAME, mode, f'{type_name}_{name}.csv')


def get_file_metadata_path(data_path, mode, type_name, name):
    return Path(data_path, SUB_DIRECTORY_NAME, mode, f'{type_name}_{name}_metadata.pkl')


def _limit_sizes(hummus_data, limit_size):
    if limit_size and isinstance(limit_size, int):
        hummus_data['recipes_df'] = hummus_data['recipes_df'][:limit_size]
        hummus_data['users_df'] = hummus_data['users_df'][:limit_size]
        hummus_data['reviews_df'] = hummus_data['reviews_df'][:limit_size]
        hummus_data['ingredients_df'] = hummus_data['ingredients_df'][:limit_size]

        debug_graph = rdflib.Graph()
        for triple in list(islice(hummus_data['food_on_ontology'], limit_size)):
            debug_graph.add(triple)
        hummus_data['food_on_ontology'] = debug_graph


def _safe_str(x):
    return str(x) if pd.notna(x) else x


def _set_new_ids(rdf_nodes_df, rdf_relationships_df):
    # Clean labels by removing leading numbers, whitespaces or '-' and removing following brackets and content if they
    # include special strings
    special_strings = ['gs1', 'foodex2', 'us cfr', 'eurofir', 'ccpr', 'codex', 'eurocode']
    special_strings_pattern = '|'.join(special_strings)
    regex_str = rf'^\d+[-\s]*|^\s*obsolete:\s*|\s*-*\s*\(([^)]*({special_strings_pattern})[^)]*)\)|-*\s$'
    rdf_nodes_df['label'] = rdf_nodes_df['label'].str.replace(regex_str, '', regex=True)

    # Create new ids for nodes for unique urls
    unique_urls = rdf_nodes_df['url'].unique()
    node_to_id = dict(zip(unique_urls, range(len(unique_urls))))

    # Set node id
    rdf_nodes_df['id'] = rdf_nodes_df['url'].map(node_to_id)

    # Update all relationships with the new labels
    rdf_relationships_df['src_id'] = rdf_relationships_df['src_id'].map(node_to_id)
    rdf_relationships_df['dst_id'] = rdf_relationships_df['dst_id'].map(node_to_id)

    return rdf_nodes_df, rdf_relationships_df, node_to_id


# Flatten_types is a dict with relationship_types mapped to the node type (or None if should be ignored)
def _flatten_kg(rdf_nodes_df, rdf_relationships_df, flatten_types_dict):
    # Get and remove all relationships with the relationship_type given in flatten_relationship_types
    flatten_relationship_types = set(flatten_types_dict.keys())
    mask_to_remove = rdf_relationships_df['relationship_type'].isin(flatten_relationship_types)
    removed_relationships = rdf_relationships_df[mask_to_remove]
    rdf_relationships_df = rdf_relationships_df[~mask_to_remove]

    # Dictionary to keep track of which nodes should be updated with new features
    node_updates = {feature_name: {} for feature_name in flatten_types_dict.values() if feature_name is not None}

    # Populate the node_updates dictionary
    for relationship_type, feature_name in flatten_types_dict.items():
        if feature_name is not None:
            relevant_relationships = removed_relationships[
                removed_relationships['relationship_type'] == relationship_type]
            for _, row in relevant_relationships.iterrows():
                if str(row['src_id']).startswith(FOOD_ON_PREFIX):
                    node_updates[feature_name][row['src_id']] = row['dst_id']

    # Apply the updates to the nodes
    for feature_name, updates in node_updates.items():
        rdf_nodes_df.loc[rdf_nodes_df['url'].isin(updates.keys()), feature_name] = rdf_nodes_df['url'].map(
            updates)

    # Remove all nodes of given type in flatten_types_dict
    rdf_nodes_df = rdf_nodes_df[~rdf_nodes_df['node_type'].isin(flatten_types_dict.values())]

    return rdf_nodes_df, rdf_relationships_df, node_updates


def _filter_relationship(relationship, node_ids):
    s, p, o = relationship
    return s in node_ids and o in node_ids


def _preprocess_food_on(graph):
    # Filter the FoodOn ontology
    filtered_triples = _filter_food_on(graph)

    # Construct a dictionary to map all subjects and object to their node type
    node_types_dict, relationship_types_dict = _map_nodes_and_relationships(filtered_triples)

    # Convert to a new graph if needed
    filtered_graph = Graph()
    # filtered_graph.addN((s, p, o, None) for s, p, o in filtered_triples) # Does not work somehow...
    for s, p, o in filtered_triples:
        filtered_graph.add((s, p, o))

    return filtered_graph, node_types_dict, relationship_types_dict


def _filter_food_on(graph):
    # Filter and collect the desired triples
    filtered_triples = []

    for s, p, o in graph:
        # Only allows triples with subjects in the FOOD_ON domain and predicates in the RELEVANT_PREDICATES set. Also,
        # must not contain a Blank Node object
        if str(s).startswith(FOOD_ON_PREFIX) and p in RELEVANT_PREDICATES and not isinstance(o, BNode):
            # For triples using the SUB_PROPERTY_OF_PREDICATE, only allow if the object is in the FOOD_ON domain
            if p == SUB_PROPERTY_OF_PREDICATE and o.startswith(FOOD_ON_PREFIX):
                filtered_triples.append((s, p, o))
            elif p == TYPE_PREDICATE and (
                    o == OWL_CLASS or o == OWL_OBJECT_PROPERTY or o == OWL_SYMMETRIC_PROPERTY):
                # Only allow if the object is an OWL Class or OWL Object Property
                filtered_triples.append((s, p, o))
            elif p != TYPE_PREDICATE and p != SUB_PROPERTY_OF_PREDICATE:  # For all other predicates
                filtered_triples.append((s, p, o))

    return filtered_triples


def _map_nodes_and_relationships(filtered_triples):
    node_types_dict = {}
    relationships_types_dict = _init_relationships_types()

    for s, p, o in filtered_triples:
        # Labels and similar
        if p == LABEL_PREDICATE and isinstance(o, Literal):  # A label
            node_types_dict[o] = 'label'
        elif p == BROAD_SYNONYM_PREDICATE and isinstance(o, Literal):  # A broad synonym
            node_types_dict[o] = 'broad_synonym'
        elif p == NARROW_SYNONYM_PREDICATE and isinstance(o, Literal):  # A narrow synonym
            node_types_dict[o] = 'narrow_synonym'
        elif p == EXACT_SYNONYM_PREDICATE and isinstance(o, Literal):  # An exact synonym
            node_types_dict[o] = 'exact_synonym'
        elif p == SYNONYM_PREDICATE and isinstance(o, Literal):  # A synonym
            node_types_dict[o] = 'synonym'
        elif p == RELATED_SYNONYM_PREDICATE and isinstance(o, Literal):  # A related synonym
            node_types_dict[o] = 'related_synonym'
        elif p == COMMENT_PREDICATE and isinstance(o, Literal):  # A comment
            node_types_dict[o] = 'comment'

        # Food On nodes
        # Redundant:
        # if str(s).startswith(FOOD_ON_PREFIX) and p == SUB_CLASS_OF_PREDICATE and str(o).startswith(
        #         FOOD_ON_PREFIX):
        #     # Subclass properties of the ontology --> Both nodes are Ingredient categories
        #     node_types_dict[s] = 'Ingredient_Type'
        #     node_types_dict[o] = 'Ingredient_Type'
        # elif str(s).startswith(FOOD_ON_PREFIX) and p == TYPE_PREDICATE and o == OWL_CLASS:
        #     node_types_dict[s] = 'Ingredient_Type'
        if str(s).startswith(FOOD_ON_PREFIX) or str(o).startswith(FOOD_ON_PREFIX):
            # Anyway, all FOOD_ON_ nodes are Ingredient_Type
            if str(s).startswith(FOOD_ON_PREFIX):
                node_types_dict[s] = 'Ingredient_Type'
            if str(o).startswith(FOOD_ON_PREFIX):
                node_types_dict[o] = 'Ingredient_Type'

    # For debugging purposes:
    # Breakpoint condition subjects: not str(s).endswith('1303') and not str(s).endswith('1563') and not str(s).endswith('4083') and not str(s).endswith('2420') and not str(s).endswith('1301') and not str(s).endswith('1560')
    # Breakpoint condition predicates: not p in OTHER_PREDICATES and p != SUB_PROPERTY_OF_PREDICATE
    # Breakpoint condition objects: not o == OWL_CLASS and not o == OWL_OBJECT_PROPERTY and not o == OWL_SYMMETRIC_PROPERTY and not str(o).startswith('http://purl.obolibrary.org/obo/NCBITaxon_') and not str(o).startswith('http://purl.obolibrary.org/obo/PO_') and not str(o).startswith('http://purl.obolibrary.org/obo/ENVO_') and not str(o).startswith('http://purl.obolibrary.org/obo/PATO_') and not str(o).startswith('http://purl.obolibrary.org/obo/GAZ_') and not str(o).startswith('http://purl.obolibrary.org/obo/GENEPIO_') and not str(o).startswith('http://purl.obolibrary.org/obo/CHEBI_') and not str(o).startswith('http://purl.obolibrary.org/obo/OBI_') and not str(o).startswith('http://purl.obolibrary.org/obo/IAO_') and not str(o).startswith('http://purl.obolibrary.org/obo/BTO_') and not str(o).endswith('1303') and not str(o).endswith('4083') and not str(o).startswith('http://purl.obolibrary.org/obo/IAO_') and not str(o).startswith('http://purl.obolibrary.org/obo/FLOPO_') and not str(o).startswith('http://purl.obolibrary.org/obo/BFO_') and not str(o).startswith('http://purl.obolibrary.org/obo/GEO_') and not str(o).startswith('http://purl.obolibrary.org/obo/NCIT_') and not str(o).startswith('http://purl.obolibrary.org/obo/BTO_')

    # Add the rest of the subjects and objects with exception type
    for s, p, o in filtered_triples:
        if s not in node_types_dict:
            node_types_dict[s] = 'exception'
        if p not in relationships_types_dict:
            relationships_types_dict[p] = 'exception'
        if o not in node_types_dict:
            node_types_dict[o] = 'exception'

    return node_types_dict, relationships_types_dict


def _init_relationships_types():
    relationships_types_dict = {
        TYPE_PREDICATE: 'has_type',
        SUB_CLASS_OF_PREDICATE: 'sub_class_of',
        LABEL_PREDICATE: 'has_label',
        COMMENT_PREDICATE: 'has_comment',
        BROAD_SYNONYM_PREDICATE: 'has_broad_synonym',
        NARROW_SYNONYM_PREDICATE: 'has_narrow_synonym',
        EXACT_SYNONYM_PREDICATE: 'has_exact_synonym',
        SYNONYM_PREDICATE: 'has_synonym',
        RELATED_SYNONYM_PREDICATE: 'has_related_synonym',
    }

    return relationships_types_dict


def _connect_kgs(ingredients_nodes, food_kg_dict, food_on_dict, node_to_id):
    # Clone the ingredients nodes
    ingredients = ingredients_nodes.copy()

    # Map ingredient ids to food_on nodes (given by urls)
    ingredients['food_kg_url'] = ingredients['ingredient_id'].map(food_kg_dict)

    # Map ingredient ids to food_on nodes (given by urls)
    ingredients['food_on_url'] = ingredients['food_kg_url'].map(food_on_dict)

    # Map food_on nodes to their labels
    ingredients['food_on_id'] = ingredients['food_on_url'].map(node_to_id)

    # Remove all non-mapped ingredients (ingredients with a None food_on_id)
    ingredient_food_on_relationships = ingredients.dropna(subset=["food_on_id"])
    ingredient_food_on_relationships['food_on_id'] = ingredient_food_on_relationships['food_on_id'].astype(int)

    ingredient_food_on_relationships['relationship_type'] = 'has_food_on'
    ingredient_food_on_relationships = ingredient_food_on_relationships[
        ['ingredient_id', 'food_on_id', 'relationship_type']]

    return ingredient_food_on_relationships


def make_unique_ids(df, id_column='id', name_column='name', id_function=None, id_max_length=256):
    """
    Modifies a DataFrame in place to replace the name_column with unique IDs. Also returns a dictionary mapping
    the old IDs (from id_column) to the new ones (the modified name_column).

    Parameters:
    :param df: The input DataFrame.
    :param id_column: The column containing the old IDs.
    :param name_column: The column to be replaced/modified with unique IDs.
    :param id_function: A function to generate the new unique IDs. If None, the default function is used.
    :param id_max_length: The maximum length of the new IDs.
    :return:  A dictionary mapping old IDs to new unique IDs.
    """
    if df is None:
        return None

    # Ensure the input DataFrame has the required columns
    if id_column not in df.columns or name_column not in df.columns:
        raise ValueError(
            f"The DataFrame must contain specified id_column ({id_column}) and name_column ({name_column}).")

    if id_function is None:
        # Use the name column to generate initial unique IDs
        df['temp_id'] = df[name_column]
    else:
        # Use the custom id function to generate initial unique IDs
        df['temp_id'] = df.apply(id_function, axis=1)

    # For all ids that are too long, shorten them. Magic number 10 is the length of the suffix that might be added to
    # ensure uniqueness. If the id is toolong cut it and add ... in the end
    magic_length = id_max_length - 10
    df['temp_id'] = df['temp_id'].apply(lambda x: x[:magic_length] + '...' if len(x) > magic_length else x)

    # Ensure uniqueness by adding suffixes to duplicates
    df['temp_id'] = (df['temp_id'] + (df.groupby('temp_id').cumcount().astype(int) + 1).astype(str)
                     .radd(' (').add(')').replace(r' \(1\)$', '', regex=True))

    # Create the mapping dictionary (old ID -> name id)
    id_name_dict = dict(zip(df[id_column], df['temp_id']))

    # Replace the old IDs with the new ones
    df[id_column] = df['temp_id']

    # Drop the temporary column
    df.drop(columns=['temp_id'], inplace=True)

    # Drop the old name column
    if id_column != name_column:
        df.drop(columns=[name_column], inplace=True)

    return id_name_dict


def apply_unique_ids(df, id_dict, id_column='id'):
    """
    Modifies a DataFrame in place to replace the old IDs with the new ones given in a dictionary.

    Parameters:
    :param df: The input DataFrame.
    :param id_dict: A dictionary mapping old IDs to new unique IDs.
    :param id_column: The column containing the old IDs.
    """
    if df is None or id_dict is None:
        return

    # Ensure the input DataFrame has the required columns
    if id_column not in df.columns:
        raise ValueError(f"The DataFrame must contain specified id_column ({id_column}).")

    # Replace the old IDs with the new ones
    df[id_column] = df[id_column].map(id_dict)


def _get_usda(hummus_data, **kwargs):
    if kwargs.get('usda', None) is not None and kwargs['usda'].get('nutrition_features', None) is not None:
        # Remove all features from usda_df not given in kwargs['usda']['nutrition_features']
        feature_dict = hummus_util.invert_dict(usda_fields.usda_rename_dict)
        nutrient_columns = [feature_dict[feature] for feature in kwargs['usda']['nutrition_features']]
    else:
        nutrient_columns = usda_fields.nutrient_columns

    # Get nutrient information for the ingredients from the usda dataset
    usda_df = hummus_data['usda_df'][['dataType'] + usda_fields.id_columns + usda_fields.text_columns +
                                     nutrient_columns]

    # Filter out all rows with are not of type 'SR Legacy' (which is the sub dataset the mappings lead to)
    usda_df = usda_df[usda_df['dataType'] == 'SR Legacy']

    def _safe_str(x):
        try:
            return ast.literal_eval(str(x))['description']
        except KeyError:
            return ast.literal_eval(str(x))
        except ValueError:
            return str(x)

    # Serialize all values in 'foodCategory' to string
    usda_df['foodCategory'] = usda_df['foodCategory'].apply(lambda x: _safe_str(x))

    # Clean data: all ingredients per default have 0.0 values for all nutrients (only columns with unit), and the
    # text fields an empty string
    usda_df[usda_fields.id_columns] = usda_df[usda_fields.id_columns].astype(int)
    usda_df[usda_fields.text_columns] = usda_df[usda_fields.text_columns].replace(np.nan, '').astype(str)
    for nutrient_column in nutrient_columns:
        usda_df[nutrient_column] = usda_df[nutrient_column].apply(lambda x: float(x) if pd.notnull(x) else x)
    usda_df = usda_df.fillna(np.nan).replace([np.nan], [None])

    # Rename columns
    usda_df.rename(columns=usda_fields.usda_rename_dict, inplace=True)
    id_columns = [usda_fields.usda_rename_dict.get(x, x) for x in usda_fields.id_columns]
    text_columns = [usda_fields.usda_rename_dict.get(x, x) for x in usda_fields.text_columns]
    nutrient_columns = [usda_fields.usda_rename_dict.get(x, x) for x in nutrient_columns]

    return id_columns, text_columns, nutrient_columns, usda_df


def _convert_to_df(pp_graph, node_types_dict, relationship_types_dict):
    # Extract nodes and relationships from RDF graph
    rdf_nodes = set()
    rdf_relationships = []

    # Collect subjects and objects
    subjects_objects = ((s, o) for s, _, o in pp_graph)
    rdf_nodes.update(chain.from_iterable(subjects_objects))

    # Collect relationships
    rdf_relationships.extend((s, p, o) for s, p, o in pp_graph)

    # Create DataFrame for RDF nodes
    rdf_nodes_df = pd.DataFrame(list(rdf_nodes), columns=['url'])
    rdf_nodes_df['node_type'] = rdf_nodes_df['url'].map(node_types_dict)

    # Remove all nodes with the node_type 'exception'
    rdf_nodes_df = rdf_nodes_df[rdf_nodes_df['node_type'] != 'exception']

    # Filter relationships such that only relationships with both subject and object in the node_urls are kept, parallelized
    # rdf_relationships = [(s, p, o) for s, p, o in rdf_relationships if s in node_ids and o in node_ids]
    node_urls = set(rdf_nodes_df['url'].values)
    with multiprocessing.Pool(processes=max(1, multiprocessing.cpu_count() // 2)) as pool:  # Use half of cores
        filtered_relationships = pool.starmap(_filter_relationship,
                                              [(relationship, node_urls) for relationship in rdf_relationships])

    # Keep only relationships passed the filter
    rdf_relationships = [relationship for relationship, keep in zip(rdf_relationships, filtered_relationships) if keep]

    # Create DataFrame for RDF relationships
    rdf_relationships_df = pd.DataFrame(rdf_relationships, columns=['src_id', 'relationship_type', 'dst_id'])
    rdf_relationships_df['relationship_type'] = rdf_relationships_df['relationship_type'].map(
        relationship_types_dict)

    # Remove all relationships with the relationship_type 'exception'
    rdf_relationships_df = rdf_relationships_df[rdf_relationships_df['relationship_type'] != 'exception']

    # Flatten nodes like e.g. comment node is not needed structurally, instead append its content to the
    # ingredient_type node
    rdf_nodes_df, rdf_relationships_df, node_updates = _flatten_kg(rdf_nodes_df, rdf_relationships_df,
                                                                   flatten_types_dict={'has_type': None,
                                                                                       'has_label': 'label',
                                                                                       'has_comment': 'comment',
                                                                                       'has_broad_synonym': 'broad_synonym',
                                                                                       'has_narrow_synonym': 'narrow_synonym',
                                                                                       'has_exact_synonym': 'exact_synonym',
                                                                                       'has_synonym': 'synonym',
                                                                                       'has_related_synonym': 'related_synonym'})

    # Stringify all cells in rdf_nodes_df and rdf_relationships_df
    rdf_nodes_df = rdf_nodes_df.applymap(_safe_str)
    rdf_nodes_df = rdf_nodes_df.astype('string')
    rdf_relationships_df = rdf_relationships_df.applymap(_safe_str)
    rdf_relationships_df = rdf_relationships_df.astype('string')

    # Set new ids
    rdf_nodes_df, rdf_relationships_df, node_to_id = _set_new_ids(rdf_nodes_df, rdf_relationships_df)

    # Change the order of the columns in rdf_nodes_df
    rdf_nodes_df = rdf_nodes_df[['id', 'label', 'node_type', 'comment', 'broad_synonym', 'narrow_synonym', 'synonym',
                                 'exact_synonym', 'related_synonym']]
    # Replace all nan values with default values given in a dictionary
    default_values = {
        'label': '',
        'comment': '',
        'broad_synonym': '',
        'narrow_synonym': '',
        'synonym': '',
        'exact_synonym': '',
        'related_synonym': ''
    }
    rdf_nodes_df = rdf_nodes_df.fillna(value=default_values)
    rdf_nodes_df = rdf_nodes_df.fillna(np.nan).replace([np.nan], [None])

    # Change the order of the columns in rdf_relationships_df
    rdf_relationships_df = rdf_relationships_df[['src_id', 'dst_id', 'relationship_type']]

    return node_to_id, rdf_nodes_df, rdf_relationships_df


def transform_to_node_df(df: pd.DataFrame, embeddings: List[float] = None) -> pd.DataFrame:
    """
    Converts DataFrame into node DataFrame formatted for Neo4j, automatically handling categorical columns.

    :param pd.DataFrame df: DataFrame to convert.
    :param List[float] embeddings: List of embeddings for the nodes, or none if no embeddings are available.
    :return: Transformed DataFrame ready for neo4j ingestion.
    :rtype: pd.DataFrame
    """
    # Adding embeddings
    if embeddings is not None:
        if 'features' in df.columns:
            print("Warning: 'features' column (== embeddings) already exists in DataFrame. Overwriting embeddings "
                  "column.")

        df['features'] = embeddings

    # Reset index
    df = df.reset_index(drop=True)

    return df


def _apply_encoding(df, category_mappings):
    """Applies categorical encodings to a DataFrame based on the given mappings.

    :param pd.DataFrame df: The DataFrame to update.
    :param dict category_mappings: Dictionary mapping column names to encoding mappings.
    """
    return df.assign(**{col: df[col].map(mapping) for col, mapping in category_mappings.items()})


def transform_to_relation_df(df: pd.DataFrame, source_col: str, destination_col: str) -> pd.DataFrame:
    """
    Converts a DataFrame into relation DataFrame(s) for Neo4j.

    :param pd.DataFrame df: Input DataFrame containing source and destination node IDs.
    :param str source_col: Column name for the source node ID.
    :param str destination_col: Column name for the destination node ID.
    :return: The relations in a DataFrame.
    :rtype: pd.DataFrame
    """
    if source_col not in df.columns or destination_col not in df.columns:
        raise ValueError("Source or target column not found in DataFrame")

    relation_df = df[[source_col, destination_col]]

    # Reset index
    relation_df = relation_df.reset_index(drop=True)

    return relation_df
