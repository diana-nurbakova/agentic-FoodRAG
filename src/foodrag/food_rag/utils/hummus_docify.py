import json
import pandas as pd
import re
from llama_index.core import Document
from tqdm.notebook import tqdm
from typing import List

from foodrag.utils import hummus_util


def recipes_to_docs(recipes_df, add_type=True, verbose=True) -> List[Document]:
    recipe_docs = data_to_docs(
        recipes_df,
        id_column="recipe_id",
        doc_type= "recipe" if add_type else None,
        rename_columns={'duration': 'duration [min]', 'ingredients': 'ingredient_quantities',
                        'ingredient_names': 'ingredients', 'direction_size': 'number_of_directions',
                        'ingredients_sizes': 'number_of_ingredients'},
        drop_columns=['normalization_comment'],
        string_columns=['title', 'description', 'recipe_url', 'servingsPerRecipe', 'serves'],
        number_columns=['duration [min]', 'average_rating', 'number_of_ratings', 'servingSize [g]', 'calories [kcal]',
                        'caloriesFromFat [kcal]', 'totalFat [g]', 'saturatedFat [g]', 'cholesterol [mg]', 'sodium [mg]',
                        'totalCarbohydrate [g]', 'dietaryFiber [g]', 'sugars [g]', 'protein [g]',
                        'number_of_directions', 'number_of_ingredients', 'who_score', 'fsa_score', 'nutri_score'],
        list_columns=['directions', 'tags', 'ingredients'],
        date_columns=['last_changed_date'],
        included_content_features=['type', 'title', 'description', 'serves', 'directions', 'ingredients',
                                   'ingredient_quantities', 'tags', 'average_rating', 'number_of_ratings',
                                   'servingSize [g]', 'calories [kcal]', 'caloriesFromFat [kcal]', 'totalFat [g]',
                                   'saturatedFat [g]', 'cholesterol [mg]', 'sodium [mg]', 'totalCarbohydrate [g]',
                                   'dietaryFiber [g]', 'sugars [g]', 'protein [g]', 'who_score', 'fsa_score',
                                   'nutri_score'],
        content_functions={
            'directions': lambda x: '\n'.join([f"    {step + 1}. {instruction}" for step, instruction in enumerate(x)]),
            'ingredient_quantities': stringify_nested_dict,
        },
        included_llm_metadata_keys=['recipe_id', 'last_changed_date'],
        included_embed_metadata_keys=['recipe_id', 'last_changed_date'],
        verbose=verbose
    )

    return recipe_docs


def users_to_docs(user_df, add_type=True, verbose=True) -> List[Document]:
    user_docs = data_to_docs(
        user_df,
        id_column="user_id",
        doc_type="user" if add_type else None,
        rename_columns={'member_id': 'user_id', 'member_url': 'url', 'member_name': 'name',
                        'member_description': 'description', 'member_status': 'status',
                        'member_avg_rating': 'avg_rating', 'member_map_pin': 'map_pin', 'member_joined': 'joined_at'},
        drop_columns=[],
        string_columns=['url', 'name', 'description', 'status', 'map_pin', 'avatar_url'],
        number_columns=['avg_rating', 'follows_count', 'follow_me_count'],
        list_columns=[],
        date_columns=['joined_at'],
        included_content_features=['type', 'name', 'description', 'status', 'map_pin'],
        content_functions={'description': lambda x: str(x) if x is not None and x != '' else None,
                           'map_pin': lambda x: str(x) if x is not None and x != '' else None},
        included_llm_metadata_keys=['user_id'],
        included_embed_metadata_keys=['user_id'],
        verbose=verbose
    )

    return user_docs


def reviews_to_docs(reviews_df, add_type=True, verbose=True) -> List[Document]:
    review_docs = data_to_docs(
        reviews_df,
        id_column=None,  # We just take the index as the ID, is automatically added and referenced as 'id'
        doc_type="review" if add_type else None,
        rename_columns={'id': 'review_id', 'member_id': 'user_id'},
        string_columns=['text'],
        number_columns=['rating', 'likes'],
        list_columns=[],
        date_columns=['last_modified_date'],
        included_content_features=['review_id', 'recipe_id', 'user_id', 'rating', 'text', 'likes'],
        content_functions={'text': lambda x: str(x) if x is not None and x != '' else None, },
        included_llm_metadata_keys=['review_id'],
        included_embed_metadata_keys=['review_id'],
        verbose=verbose
    )

    return review_docs


def ingredients_to_docs(ingredients_df, add_type=True, verbose=True) -> List[Document]:
    ingredient_docs = data_to_docs(
        ingredients_df,
        id_column="ingredient_id",
        doc_type="ingredient" if add_type else None,
        rename_columns={'ing_name': 'name'},
        drop_columns=[],
        string_columns=['name'],
        number_columns=[],
        list_columns=[],
        date_columns=[],
        included_content_features=['type', 'name', 'ingredient_id', 'recipe_id'],
        content_functions={},
        included_llm_metadata_keys=['ingredient_id'],
        included_embed_metadata_keys=['ingredient_id'],
        verbose=verbose
    )

    return ingredient_docs


def food_products_to_docs(usda_df, add_type=True, verbose=True) -> List[Document]:
    food_product_docs = data_to_docs(
        usda_df,
        id_column=None,
        doc_type="food_product" if add_type else None,
        rename_columns={'id': 'food_product_id', 'ndbNumber': 'usda_id', 'description': 'name',
                        'brandOwner': 'brand_owner', 'servingSize': 'serving_size',
                        'servingSizeUnit': 'serving_size_unit', 'brandedFoodCategory': 'food_category',
                        'total lipid (fat) [g]': 'total fat [g]', 'carbohydrate, by difference [g]': 'carbohydrate [g]',
                        'fiber, total dietary [g]': 'fiber [g]', 'calcium, ca [mg]': 'calcium [mg]',
                        'iron, fe [mg]': 'iron [mg]', 'sodium, na [mg]': 'sodium [mg]',
                        'vitamin c, total ascorbic acid [mg]': 'vitamin c [mg]',
                        'fatty acids, total trans [g]': 'trans fat [g]',
                        'fatty acids, total saturated [g]': 'saturated fat [g]'},
        drop_columns=[],
        string_columns=['name', 'brand_owner', 'ingredients', 'serving_size_unit', 'food_category'],
        number_columns=['serving_size', 'protein [g]', 'total fat [g]', 'carbohydrate [g]', 'energy [kcal]',
                        'total sugars [g]', 'fiber [g]', 'calcium [mg]', 'iron [mg]', 'sodium [mg]', 'vitamin c [mg]',
                        'cholesterol [mg]', 'trans fat [g]', 'saturated fat [g]'],
        list_columns=[],
        date_columns=[],
        included_content_features=['type', 'name', 'food_product_id', 'usda_id', 'serving_size', 'serving_size_unit',
                                   'food_category', 'protein [g]', 'total fat [g]', 'carbohydrate [g]',
                                   'energy [kcal]', 'total sugars [g]', 'fiber [g]', 'calcium [mg]', 'iron [mg]',
                                   'sodium [mg]', 'vitamin c [mg]', 'cholesterol [mg]', 'trans fat [g]',
                                   'saturated fat [g]'],
        content_functions={'name': lambda s: ' '.join(word.capitalize() for word in s.lower().split())},
        included_llm_metadata_keys=['food_product_id'],
        included_embed_metadata_keys=['food_product_id'],
        verbose=verbose
    )

    return food_product_docs


def food_on_to_docs(food_on_ontology, add_type=True, verbose=True) -> List[Document]:
    # Queries for all FOODON_ nodes with at least one superclass, so we only skip the root node
    node_results = food_on_ontology.query(f"""
        SELECT DISTINCT ?node ?label ?super_node_label ?sub_node_label
        WHERE {{
            ?node <http://www.w3.org/2000/01/rdf-schema#label> ?label .
            FILTER(CONTAINS(STR(?node), "FOODON_"))
        
            ?node <http://www.w3.org/2000/01/rdf-schema#subClassOf> ?super_node .
            ?super_node <http://www.w3.org/2000/01/rdf-schema#label> ?super_node_label .
        
            OPTIONAL {{
                ?sub_node <http://www.w3.org/2000/01/rdf-schema#subClassOf> ?node .
                ?sub_node <http://www.w3.org/2000/01/rdf-schema#label> ?sub_node_label .
            }}
        }}
        """)

    # Process the results into a dictionary to aggregate superclasses and subclasses
    node_info_dict = {}
    for binding in node_results:
        node = str(binding["node"])
        label = str(binding["label"]) if binding["label"] else None
        super_node_label = str(binding["super_node_label"]) if binding["super_node_label"] else None
        sub_node_label = str(binding["sub_node_label"]) if binding["sub_node_label"] else None

        # Initialize entry for the node if not already present
        if node not in node_info_dict:
            node_info_dict[node] = {
                "node": node,
                "name": label,
                "super_nodes": [],
                "sub_nodes": [],
            }

        # Add superclasses and subclasses
        if super_node_label and super_node_label not in node_info_dict[node]["super_nodes"]:
            node_info_dict[node]["super_nodes"].append(super_node_label)
        if sub_node_label and sub_node_label not in node_info_dict[node]["sub_nodes"]:
            node_info_dict[node]["sub_nodes"].append(sub_node_label)

    # Convert the dictionary to a DataFrame
    food_on_df = pd.DataFrame.from_records(
        list(node_info_dict.values()),
        columns=["node", "name", "super_nodes", "sub_nodes"]
    )

    # Convert the DataFrame to documents
    food_on_docs = data_to_docs(
        food_on_df,
        id_column=None,
        doc_type="food_on_product" if add_type else None,
        rename_columns={'id': 'food_on_id'},
        string_columns=['name'],
        number_columns=[],
        list_columns=['super_nodes', 'sub_nodes'],
        date_columns=[],
        included_content_features=['name', 'super_nodes', 'sub_nodes'],
        content_functions={},
        included_llm_metadata_keys=[],
        included_embed_metadata_keys=[],
        verbose=verbose
    )

    return food_on_docs


def data_to_docs(df, id_column=None, included_content_features=[], content_functions={}, included_llm_metadata_keys=[],
                 included_embed_metadata_keys=[], rename_columns={}, drop_columns=[], string_columns=[],
                 number_columns=[], list_columns=[], date_columns=[], string_default='', number_default=0,
                 date_default='2000-01-01', doc_type=None,
                 verbose=True) -> List[Document]:
    """
    Converts a dataframe into a list of document objects.

    :param pd.DataFrame df: The dataframe to process.
    :param str id_column: The column to use as the document ID (optional).
    :param list included_content_features: Columns to include in the document content.
    :param dict content_functions: Custom formatting functions for specific columns.
    :param list included_llm_metadata_keys: Metadata keys to include for LLM processing.
    :param list included_embed_metadata_keys: Metadata keys to include for embeddings.
    :param dict rename_columns: Mapping of columns to rename.
    :param list drop_columns: Columns to drop from the dataframe.
    :param list string_columns: List of columns to process as strings.
    :param list number_columns: List of columns to process as numbers.
    :param list list_columns: List of columns to process as lists.
    :param list date_columns: List of columns to process as dates.
    :param str string_default: Default value to use for string columns (default is '').
    :param int number_default: Default value to use for number columns (default is 0).
    :param str date_default: Default value for date columns (default is '2000-01-01').
    :param str doc_type: Optional type to add as a column (e.g., "recipe" or "user").
    :param bool verbose: Whether to show a progress bar.

    :returns: List of document objects.
    :rtype: list[Document]
    """
    documents = []

    # Clean the data
    df, id_column, included_content_features = clean_df(df=df, id_column=id_column,
                                                        included_content_features=included_content_features,
                                                        content_functions={},  # functions used at document creation
                                                        rename_columns=rename_columns,
                                                        drop_columns=drop_columns,
                                                        string_columns=string_columns,
                                                        number_columns=number_columns, list_columns=list_columns,
                                                        date_columns=date_columns, string_default=string_default,
                                                        number_default=number_default, date_default=date_default,
                                                        doc_type=doc_type,
                                                        stringify_columns=[],  # stringify is done in next step instead
                                                        drop_in_df=False)

    # Define excluded metadata keys
    excluded_llm_metadata_keys = df.columns.difference(included_llm_metadata_keys).tolist()
    excluded_embed_metadata_keys = df.columns.difference(included_embed_metadata_keys).tolist()

    # Create the documents
    for index, row in tqdm(df.iterrows(), total=df.shape[0],
                           desc=f"Creating {doc_type + ' ' if doc_type else ''}documents", disable=not verbose):
        # Create content with optional custom functions
        content = _stringify_row(row, included_content_features=included_content_features,
                                 feature_functions=content_functions)

        # Append the document
        documents.append(Document(id_=str(row.get(id_column, index)), text=content, metadata=row.to_dict(),
                                  excluded_llm_metadata_keys=excluded_llm_metadata_keys,
                                  excluded_embed_metadata_keys=excluded_embed_metadata_keys))

    return documents


def clean_df(df, id_column=None, included_content_features=[], content_functions={}, rename_columns=[], drop_columns=[],
             string_columns=[], number_columns=[], list_columns=[], date_columns=[], string_default='',
             number_default=0, date_default='2000-01-01',
             doc_type=None, stringify_columns=[], drop_in_df=True) -> (pd.DataFrame, str, list):
    """
    Cleans a dataframe.

    :param pd.DataFrame df: The dataframe to process.
    :param str id_column: The column to use as the document ID (optional).
    :param list included_content_features: Columns to include in the document content.
    :param dict content_functions: Custom formatting functions for specific columns.
    :param dict rename_columns: Mapping of columns to rename.
    :param list drop_columns: Columns to drop from the dataframe.
    :param list string_columns: List of columns to process as strings.
    :param list number_columns: List of columns to process as numbers.
    :param list list_columns: List of columns to process as lists.
    :param list date_columns: List of columns to process as dates.
    :param str string_default: Default value to use for string columns (default is '').
    :param int number_default: Default value to use for number columns (default is 0).
    :param str date_default: Default value for date columns (default is '2000-01-01').
    :param str doc_type: Optional type to add as a column (e.g., "recipe" or "user").
    :param list stringify_columns: Columns to stringify. Columns are processed as strings after all other functions are
    applied.
    :param bool drop_in_df: Whether to drop the columns in the dataframe or not. (For backwards compatibility turn set
     to False)

    :returns: Triple consisting of: Dataframe with cleaned data, ID column name, and included content features.
    :rtype: (pd.DataFrame, str, list)
    """
    # Optionally add a type column
    if doc_type:
        df['type'] = doc_type

        if 'type' not in included_content_features:
            included_content_features = ['type'] + included_content_features

    # Add the ID column if not already included
    if id_column is None:
        df['id'] = df.index
        id_column = rename_columns.get('id', 'id')

    # Make sure that the ID column is the first column
    if id_column not in included_content_features:
        included_content_features = [id_column] + included_content_features

    # Clean the data
    df = _clean_data(df=df,
                     content_functions=content_functions,
                     rename_columns=rename_columns,
                     drop_columns=drop_columns,
                     string_columns=string_columns,
                     number_columns=number_columns,
                     list_columns=list_columns,
                     date_columns=date_columns,
                     string_default=string_default,
                     number_default=number_default,
                     date_default=date_default,
                     stringify_columns=stringify_columns)

    # Make sure that all included_content_features are in df, and remove those that are not
    included_content_features = [col for col in included_content_features if col in df.columns]

    if drop_in_df:
        # Update column positions and drop columns not in included_content_features
        df = df[included_content_features]
    else:
        # Update (all) column positions
        df = df[included_content_features + df.columns.difference(included_content_features).tolist()]

    return df, id_column, included_content_features


def triple_data_to_doc(df, included_content_features=[], content_functions={}, rename_columns={}, drop_columns=[],
                       string_columns=[], number_columns=[], list_columns=[], date_columns=[], string_default='',
                       number_default=0, date_default='2000-01-01', pre_string='', post_string='', doc_type=None):
    """
    Converts a dataframe into a single document object, where each row is a line in the document.

    :param pd.DataFrame df: The dataframe to process.
    :param list included_content_features: Columns to include in the document content.
    :param dict content_functions: Custom formatting functions for specific columns.
    :param dict rename_columns: Mapping of columns to rename.
    :param list drop_columns: Columns to drop from the dataframe.
    :param list string_columns: List of columns to process as strings.
    :param list number_columns: List of columns to process as numbers.
    :param list list_columns: List of columns to process as lists.
    :param list date_columns: List of columns to process as dates.
    :param str string_default: Default value to use for string columns (default is '').
    :param int number_default: Default value to use for number columns (default is 0).
    :param str date_default: Default value for date columns (default is '2000-01-01').
    :param str pre_string: String to add at the beginning of the document.
    :param str post_string: String to add at the end of the document.
    :param str doc_type: Optional type to add to the process bar.

    :returns: A single document object containing all rows of the dataframe.
    :rtype: Document
    """
    # Clean the data
    df = _clean_data(df=df,
                     content_functions={},  # functions used at document creation
                     rename_columns=rename_columns,
                     drop_columns=drop_columns,
                     string_columns=string_columns,
                     number_columns=number_columns,
                     list_columns=list_columns,
                     date_columns=date_columns,
                     string_default=string_default,
                     number_default=number_default,
                     date_default=date_default,
                     stringify_columns=[])  # stringify is done in the next step instead

    # Select columns to include in the content
    content_df = df[included_content_features] if included_content_features else df

    # Create content for the document
    rows = []
    for _, row in tqdm(content_df.iterrows(), total=content_df.shape[0],
                       desc=f"Creating single {doc_type + ' ' if doc_type else ''}document by parsing multiple rows"):
        # Format row with optional custom functions
        formatted_row = " ".join(
            content_functions.get(col, str)(row[col]) if col in content_functions else str(row[col])
            for col in content_df.columns
        )
        rows.append(formatted_row)

    # Combine all rows into a single document content and add pre- and post-strings
    content = pre_string + "\n".join(rows) + post_string

    # Create the document
    return Document(text=content, metadata={})


def strings_to_docs(strings: List[str], doc_type=None) -> List[Document]:
    """
    Converts a list of strings into a list of documents.

    :param List[str] strings: List of strings to process.
    :param str doc_type: Optional type to add to the process bar.

    :returns: A list of documents.
    :rtype: List[Document]
    """
    documents = []
    for index, string in tqdm(enumerate(strings), total=len(strings),
                              desc=f"Creating {doc_type + ' ' if doc_type else ''}documents"):
        documents.append(Document(id_=str(index), text=string, metadata={'index': index}))

    return documents


def qa_question_to_docs(qa_set: dict, doc_type=None) -> List[Document]:
    """
    Converts questions of a question-answer set into a list of documents.

    :param dict qa_set: The question-answer set to process.
    :param str doc_type: Optional type to add to the process bar.

    :returns: A list of documents.
    :rtype: List[Document]
    """
    documents = []

    # Get all questions
    questions = {index: [qa['question']] for index, qa in enumerate(qa_set)}
    llm_questions = {index: [q_dict['question'] for q_dict in qa['llm_questions']]
                     for index, qa in enumerate(qa_set) if 'llm_questions' in qa}
    all_questions = hummus_util.combine_dicts(questions, llm_questions)

    for qa_index, questions in tqdm(all_questions.items(), total=len(all_questions),
                                    desc=f"Creating {doc_type + ' ' if doc_type else ''}documents"):
        for question in questions:  # Append the document for each question
            documents.append(Document(text=question, metadata={'qa_index': qa_index, 'question': question}))

    return documents


def _clean_data(df, content_functions={}, rename_columns={}, drop_columns=[], string_columns=[], number_columns=[],
                list_columns=[], date_columns=[], string_default='', number_default=0, date_default='2000-01-01',
                stringify_columns=[]):
    """
    Cleans a dataframe by renaming columns, dropping unnecessary columns, and replacing NaN values.

    :param pd.DataFrame df: The dataframe to clean.
    :param dict content_functions: Dictionary mapping column names to custom formatting functions.
    :param dict rename_columns: Dictionary mapping old column names to new column names.
    :param list drop_columns: List of columns to drop.
    :param list string_columns: Columns with string values to replace NaN with an empty string.
    :param list number_columns: Columns with numeric values to replace NaN with 0.
    :param list list_columns: Columns with list-like values to replace NaN with empty lists.
    :param list date_columns: Columns with date values to replace NaN with `date_default`.
    :param str string_default: Default value to use for string columns (default is '').
    :param int number_default: Default value to use for number columns (default is 0).
    :param str date_default: Default value to use for date columns (default is '2000-01-01').
    :param list stringify_columns: Columns to stringify. Columns are processed as strings after all other functions are
    applied.

    :returns: The cleaned dataframe.
    :rtype: pd.DataFrame
    """
    # Rename columns
    if rename_columns:
        # Get renamed columns duplicates and remove the old ones
        new_columns = [new_col for old_col, new_col in rename_columns.items() if old_col in df.columns]
        for col in new_columns:
            if col in df.columns:
                # Remove the old column from the df
                df = df.drop(columns=col, errors='ignore')

        # Rename the columns
        df = df.rename(columns=rename_columns, errors='ignore')  # ignore non-existing columns

    # Drop columns
    if drop_columns:
        df = df.drop(columns=drop_columns, errors='ignore')  # ignore already dropped columns

    # Replace string-based NaN values
    if string_columns:
        existing_string_columns = [col for col in string_columns if col in df.columns]
        df[existing_string_columns] = df[existing_string_columns].fillna(value=string_default)

    # Replace number-based NaN values
    if number_columns:
        existing_number_columns = [col for col in number_columns if col in df.columns]
        df[existing_number_columns] = df[existing_number_columns].fillna(value=number_default)

    # Replace list-based NaN values
    if list_columns:
        existing_list_columns = [col for col in list_columns if col in df.columns]
        df[existing_list_columns] = df[existing_list_columns].applymap(_to_list)

    # Replace date-based NaN values
    if date_columns:
        existing_date_columns = [col for col in date_columns if col in df.columns]
        df[existing_date_columns] = df[existing_date_columns].fillna(value=date_default)

    # User based functions
    if content_functions:
        for col, func in content_functions.items():
            if col in df.columns:
                df[col] = df[col].apply(func)

    # Stringify columns
    if stringify_columns:
        existing_stringify_columns = [col for col in stringify_columns if col in df.columns]
        df[existing_stringify_columns] = df[existing_stringify_columns].applymap(str)

    return df


def _to_list(x):
    if isinstance(x, list):
        return x
    elif isinstance(x, str):
        parts = x.split('.')
        # Remove spaces at the start/end, keep the '.' at the end, but skip empty items.
        return [part.strip() + '.' for part in parts if part.strip()]
    else:
        return []


def _get_metadata(row, features):
    return row[features].to_dict()


def _stringify_row(row, included_content_features=None, feature_functions=None):
    if included_content_features is None:
        included_content_features = list(row.keys())

    if feature_functions is None:
        feature_functions = {}

    def clean_key(k):
        # First letter to uppercase
        k = k[0].upper() + k[1:]
        # Replace underscores with spaces
        k = k.replace("_", " ")
        return k

    def format_value(feature_name, value, content_features, functions):
        if feature_name in content_features:
            if feature_name in functions:  # Format based on user-defined function
                function = functions[feature_name]
                if function is not None:
                    new_value = function(value)
                    if new_value is not None:
                        value_length = len(str(new_value))
                        if value_length > 110 or '\n' in str(new_value):
                            return f"{clean_key(feature_name)}:\n{new_value}"
                        else:
                            return f"{clean_key(feature_name)}: {new_value}"
                    else:
                        return None
                else:
                    return None
            else:  # Format based on type
                if isinstance(value, float):
                    new_value = round(value, 2)
                elif isinstance(value, (int, str)):
                    new_value = str(value)
                elif isinstance(value, enumerate):
                    new_value = "\n".join(f"    - {item}" for item in value)
                elif isinstance(value, list):
                    new_value = ", ".join(str(item) for item in value)
                else:
                    new_value = str(value)
                if new_value is not None and new_value != '':
                    value_length = len(str(new_value))
                    if value_length > 110 or '\n' in str(new_value):
                        return f"{clean_key(feature_name)}:\n{new_value}"
                    else:
                        return f"{clean_key(feature_name)}: {new_value}"
                else:
                    return None
        else:
            return None

    formatted_content = "\n".join(
        formatted_value for feature_name, value in row.items()  # automatically removes None/empty/'' values
        if (formatted_value := format_value(feature_name, value, included_content_features, feature_functions))
    )

    return formatted_content


def _clean_dict(x):
    clean_dict = dict()
    if not pd.isnull(x):
        try:
            json.dumps(x)  # ignore the result, just check if it is serializable
            clean_dict = x
        except TypeError:
            clean_dict = str(x)

    return clean_dict


def to_triples(df, id_column, current_id, subject_column, predicate, object_column, different_data_types=False):
    filtered_df = df[df[id_column] == current_id]

    if different_data_types:
        return [(r0, predicate, r1) for r0, r1 in zip(filtered_df[subject_column], filtered_df[object_column])]
    else:
        return [(row[0], predicate, row[1]) for row in filtered_df[[subject_column, object_column]].to_numpy()]


def stringify_nested_lists(df):
    return df.apply(lambda col: col.apply(_stringify_list))


def _stringify_list(cell):
    return ' '.join(
        elem.strip() if elem.strip().endswith(('.', ',', ';', '!', '?')) else f"{elem.strip()},"
        for elem in map(str, cell)
    ) if isinstance(cell, list) else cell


def stringify_nested_dicts(df):
    return df.apply(lambda col: col.apply(stringify_nested_dict))


def stringify_nested_dict(cell):
    if isinstance(cell, dict):
        result = []
        for category, items in cell.items():
            if len(cell) <= 1:
                category_name = ''  # Omit category for a single key
            elif category == '':
                category_name = 'Main Recipe: '  # Rename empty category
            else:
                category_name = f"{category.strip().title()}: "

            # Process the list of tuples
            items_str = []
            for pair in items:
                ingredient = str(pair[0]).strip()
                amount = str(pair[1]).strip()

                # Clean the amount
                amount = re.sub(r'^\s*1?\s*time\(s\)\s*$', '', amount)
                amount = re.sub(r'time\(s\)', '', amount)  # remove remaining 'time(s)'
                amount = re.sub(r'\s+', ' ', amount)  # change longer spaces to single spaces
                amount = amount.strip()

                # Combine ingredient and amount
                if amount == '':
                    element = ingredient
                else:
                    element = f"{amount} of {ingredient}"

                # Append all elements
                if element.endswith(('.', ',', ';', '!', '?')):
                    items_str.append(element)
                else:
                    items_str.append(f"{element};")

            # Combine category and items and remove the last punctuation
            category_str = category_name + ' '.join(items_str)
            if category_str.endswith((',', ';', '.', '!', '?')):
                category_str = category_str[:-1]

            # Remove old template tags
            category_str = re.sub(r' *\$template\d*\$', '', category_str)

            result.append(category_str)

        # Join all categories with newlines
        return '\n'.join(result)
    else:
        return cell
