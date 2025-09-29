import json

import pandas as pd
import yaml
from tqdm.notebook import tqdm

from src.food_rag.evaluation import recipe_scores


def get_best_questions(qa_set, limit_size: int = -1, discard_empty_qas=False, show_progress=True):
    """
    Gets the best fitting question for an answer.

    :param qa_set: The question-answer set to evaluate.
    :param int limit_size: The maximum number of question-answer pairs to return (default is -1, which means no limit).
    :param bool discard_empty_qas: Whether to discard empty question-answer pairs (default is False).
    :param bool show_progress: Whether to display a progress bar (default is True).

    :returns: The best fitting question for an answer.
    """
    converted_qa_set = {}

    if discard_empty_qas:
        if show_progress:
            print("Discarding empty question-answer pairs.")
        qa_set = [qa for qa in qa_set if not qa['answer'].empty]

    if limit_size > 0:
        if show_progress:
            print(f"Limiting the number of question-answer pairs to {limit_size}.")
        qa_set = qa_set[:limit_size]

    iterator = tqdm(qa_set, disable=not show_progress, desc="Converting QA-set")
    for i, question_answers in enumerate(iterator):
        # Try finding a high-score question
        high_score_question = next(
            (llm_q['question'] for llm_q in question_answers['llm_questions'] if llm_q.get('score', 0) >= 5),
            question_answers['question']
        )

        # Append the final question-answer pair
        converted_qa_set[i] = {
            'question': high_score_question,
            'answer': question_answers['answer']['recipe_id'].to_list(),
            'persona': question_answers['persona']
        }

    return converted_qa_set


def extend_qa_embeddings(qa_set, qa_doc_dict):
    """
    Loads the embeddings for the question-answer set.

    :param qa_set: The question-answer set to evaluate.
    :param qa_doc_dict: The dictionary containing the question-answer embeddings.

    :returns: The embeddings for the question-answer set.
    """
    # Create a dict from question string to embedding
    question_embedding_dict = {qa[0].metadata['question']: [node.embedding for node in qa] for qa in
                               qa_doc_dict.values()}
    num_loaded_embedded_questions = len(question_embedding_dict)
    print(f"Loaded embeddings for {num_loaded_embedded_questions} questions, possibly including llm_questions and also"
          f" duplicates.")

    # Remove all non-embeddings
    question_embedding_dict = {question: embeddings for question, embeddings in question_embedding_dict.items()
                               if embeddings is not None}

    # Remove questions with multiple embeddings (--> this means the question was too long to be embedded by itself and
    # got split into multiple nodes)
    question_embedding_dict = {question: embeddings[0] for question, embeddings in question_embedding_dict.items()
                               if len(embeddings) == 1}

    # Print the number of questions that were removed because they had multiple embeddings
    if len(question_embedding_dict) < num_loaded_embedded_questions:
        print(f"Removed {num_loaded_embedded_questions - len(question_embedding_dict)} questions with multiple"
              f" embeddings.")

    # Add the embeddings to the QA-set
    for _, qa in qa_set.items():
        qa['embedding'] = question_embedding_dict.get(qa['question'], None)

    return qa_set


def get_nutrition_scores(recipes_df, score_names=['nutri_score', 'who_score', 'fsa_score']):
    return recipe_scores.avg_nutrition_score(recipes_df, score_names)


def get_qa_nutrition_scores(qa_set, recipes_df, score_names=['nutri_score', 'who_score', 'fsa_score']):
    """
    Gets the nutrition scores for the QA-set.

    :param qa_set: The question-answer set to evaluate.
    :param pd.DataFrame recipes_df: The recipes dataframe containing nutritional information.
    :param list score_names: List of score names to retrieve (default is ['nutri_score', 'who_score', 'fsa_score']).

    :returns: A DataFrame with nutrition scores for the QA-set.
    :rtype: pd.DataFrame
    """
    valid_answers = [qa['answer'] for qa in qa_set.values()]
    valid_unique_answers = set(item for sublist in valid_answers for item in sublist)
    valid_unique_recipes = recipes_df[recipes_df['recipe_id'].isin(valid_unique_answers)]
    valid_nutrition_scores = valid_unique_recipes[score_names].describe().loc[['min', 'mean', 'max']]

    return valid_nutrition_scores


def recipe_id_to_df(recipe_ids, recipes_df):
    return recipes_df[recipes_df['recipe_id'].isin(recipe_ids)]


def get_recipes_from_ids(qa_set, recipes_df):
    """
    Sets the recipes from the QA-set by replacing the recipe IDs with the actual recipes.

    :param qa_set: The question-answer set to evaluate.
    :param pd.DataFrame recipes_df: The recipes dataframe containing nutritional information.

    :returns: The modified question-answer set with the actual recipes.
    """
    for qa in qa_set.values():
        qa['answer'] = recipe_id_to_df(qa['answer'], recipes_df)

    return qa_set


def get_contexts_qa_set(qa_set):
    """
    Adds the contexts from the QA-set.
    :param qa_set: The question-answer set to evaluate.
    :return: The QA-set with the references.
    """
    for qa in tqdm(qa_set.values(), desc="Adding references to QA-set"):

        # Stringify all the answers
        answers = qa['answer']
        if len(answers) == 0:
            recipe_text = 'There are valid answers.'
        elif not isinstance(answers, pd.DataFrame):
            recipe_text = ''  # When the answers are not a DataFrame, they are not added to the references
        else:
            recipe_text = ('Valid recipes (by titles):\n----------------------\n'
                           + '\n'.join(answers['title'].to_list())
                           + '\n----------------------\n')

        # Stringify the persona
        persona = qa['persona']
        persona['nutrients_per_recipe'] = persona.pop('relative_nutrients') # rename a key
        persona['minimal_nutrition_scores'] = persona.pop('nutrition_scores') # rename a key
        persona = remove_empty_values(persona)
        persona = ('User Persona:\n----------------------\n'
                    + yaml.dump(persona, sort_keys=False, default_flow_style=False)
                    + '\n----------------------\n')

        # Add the references and persona to the QA-set
        qa['recipe_titles'] = recipe_text
        qa['personas'] = persona

    return qa_set

def remove_empty_values(d):
    """Recursively remove all key-value pairs from a dict if the value is empty."""
    if not isinstance(d, dict):
        return d

    cleaned_dict = {}
    for key, value in d.items():
        # If it's a dict, process it recursively
        if isinstance(value, dict):
            nested = remove_empty_values(value)
            if nested:  # Keep only if not empty
                cleaned_dict[key] = nested
        # If it's a list, remove empty elements within it and keep if result isn't empty
        elif isinstance(value, list):
            nested_list = []
            for item in value:
                if isinstance(item, dict):
                    cleaned_item = remove_empty_values(item)
                    if cleaned_item:
                        nested_list.append(cleaned_item)
                elif item not in [None, "", []]:
                    nested_list.append(item)
            if nested_list:
                cleaned_dict[key] = nested_list
        # For other types, keep if not empty or None
        elif value not in [None, ""]:
            cleaned_dict[key] = value

    return cleaned_dict