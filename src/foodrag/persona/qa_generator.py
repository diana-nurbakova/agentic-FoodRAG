import asyncio
import json
import pandas as pd
from enum import Enum
from litellm import acompletion, OpenAIError
from pathlib import Path
from tqdm.notebook import tqdm

from foodrag.persona import persona_generator
from foodrag.utils import load_util
from foodrag.utils import rdf_util

SUB_DIRECTORY_NAME = 'qa'


# Define an Enum for template types
class QuestionType(Enum):
    RECIPE = 'recipe'
    MEAL_PLAN = 'meal_plan'

    def __str__(self):
        return self.value


# Function to generate questions and answers
def generate_question_answer_set(hummus_data, question_type: QuestionType, persona_config,
                                 nutrition_config, data_path=None, sub_dir_name=None, llm=None, size: int = 10000,
                                 batch_size: int = 10, storage_frequency: int = 1000,
                                 food_on: rdf_util.PreProcessedFoodOn = None) -> list[dict]:
    """
    Generate a set of questions and answers based on the given templates and persona.
    :param hummus_data: The Hummus data including recipes and nutrition information.
    :param question_type: The type of question to generate (recipe or meal plan).
    :param persona_config: The persona configuration.
    :param nutrition_config: The nutrition configuration.
    :param data_path: The path to the data storage directory.
    :param sub_dir_name: The name of the sub-directory (under data/qa) to store the question-answer set. If None, it
     will be stored directly in data/qa.
    :param llm: The language model to use for question generation (optional).
    :param size: Size of the question-answer set to generate (default is 10 000).
    :param batch_size: Number of concurrent requests to be handled at a time (default is 10).
    :param storage_frequency: Frequency of storing the question-answer set to a file (default is 1000).
    :param food_on: Pre-processed FoodOn data (optional).

    :return: A dict containing a question and its corresponding answer.
    """
    stage = 'simple_qa'
    question_answer_set = []
    file_name = str(question_type) + '_question_answer_set'

    if data_path:
        path = Path(data_path, SUB_DIRECTORY_NAME, sub_dir_name if sub_dir_name else '')
        try:  # Try to load the question-answer set from a file
            loaded_data = load_util.load_data(path)
            stage = loaded_data[f'stage_{str(question_type)}']
            question_answer_set = loaded_data[file_name]
            print(f"Resuming with stage '{stage}'.")
        except Exception:
            print(f"Starting stage '{stage}'.")
    else:
        print(
            "No data path provided. Skipping loading and saving of question-answer set and just generating the qa set.")
        path = None

    if stage == 'simple_qa':
        size -= len(question_answer_set)

        if food_on is None:
            food_on = rdf_util.PreProcessedFoodOn(hummus_data)

        for _ in tqdm(range(size), desc='Generating Question-Answers'):
            # 1. Generate Restrictions
            persona = persona_generator.generate_persona_preferences(persona_config, nutrition_config, hummus_data,
                                                                     food_on=food_on)
            if question_type == QuestionType.MEAL_PLAN:
                meal_plan = persona_generator.get_random_meal_plan_nutrients(persona_config)
            else:
                meal_plan = None

            # 2. Generate a question based on the persona, meal_plan_restriction, and question type
            question = _generate_question(persona, meal_plan, question_type)

            # 3. Generate a valid answer (assuming you have a method `generate_valid_answer_for_persona`)
            if question_type == QuestionType.RECIPE:
                answer = _get_valid_recipes_for_persona(food_on, persona_config, persona, hummus_data['recipes_df'])
            elif question_type == QuestionType.MEAL_PLAN:
                answer = _generate_valid_meal_plans_for_persona(food_on, persona_config, persona, meal_plan,
                                                                hummus_data['recipes_df'])
            else:
                raise ValueError(f"Invalid question type: {question_type}")

            # 4. Add the question-answer pair
            question_answer_set.append({'question': question,
                                        'llm_questions': [],
                                        'answer': answer,
                                        'persona': persona,
                                        'meal_plan': meal_plan if question_type == QuestionType.MEAL_PLAN else None})

            if len(question_answer_set) % storage_frequency == 0:
                if path:
                    _save(path, file_name, stage, question_answer_set, question_type)

        stage = 'llm_qa'
        if path:
            _save(path, file_name, stage, question_answer_set, question_type, end_stage=True)
        if llm:
            print(f"Starting stage '{stage}'.")

    # 5. Optional: Formulate the question with an LLM to make it more natural
    if stage == 'llm_qa' and llm:
        asyncio.run(
            reformulate_questions(question_answer_set, llm, path, file_name, question_type, batch_size=batch_size,
                                  storage_frequency=storage_frequency))
        stage = 'scored_qa'
        if path:
            _save(path, file_name, stage, question_answer_set, question_type, end_stage=True)
        print(f"Starting stage '{stage}'.")

    # 6. Optional: Rate the reformulated questions
    if stage == 'scored_qa' and llm:
        asyncio.run(rate_reformulated_questions(question_answer_set, llm, path, file_name, question_type,
                                                batch_size=batch_size, storage_frequency=storage_frequency))
        stage = 'done'
        if path:
            _save(path, file_name, stage, question_answer_set, question_type, end_stage=True)

    return question_answer_set


def _save(path, file_name, stage, question_answer_set, question_type, end_stage=False):
    load_util.ensure_dir(path)

    if path:  # Save the question-answer set to a file
        if end_stage:
            print(f"Finally, saving question-answer to file: '{file_name}'")
        else:
            print(f"Saving question-answer set in stage '{stage}' to file: '{file_name}'")

        load_util.store_data({'stage_' + str(question_type): stage, file_name: question_answer_set}, path)


def _generate_question(persona: dict, meal_plan_restriction: dict, question_type: QuestionType) -> str:
    """
    Generate a question based on the persona's preferences and constraints.

    :param persona: The persona with likes, dislikes, and nutrient constraints.
    :param meal_plan_restriction: The meal plan restrictions for each meal type (if applicable).
    :param question_type: Type of the question ('recipe' or 'meal_plan').

    :return: The generated question.
    """

    if question_type == QuestionType.RECIPE:
        question = "What are the recipes that meet the following criteria?"
    elif question_type == QuestionType.MEAL_PLAN:
        question = "What are the meal plans that meet the following criteria?"
    else:
        raise ValueError(f"Invalid question type: {question_type}")

    # Add tag likes to the question
    tags_likes = ', '.join(persona['tags']['likes']) if persona['tags']['likes'] else None
    if tags_likes:
        question += f"\n - Tags Liked: {tags_likes}"

    # Add tag dislikes to the question
    tags_dislikes = ', '.join(persona['tags']['dislikes']) if persona['tags']['dislikes'] else None
    if tags_dislikes:
        question += f"\n - Tags Disliked: {tags_dislikes}"

    # Add ingredients likes to the question
    ingredients_likes = ', '.join(persona['ingredients']['likes']) if persona['ingredients']['likes'] else None
    if ingredients_likes:
        question += f"\n - Ingredients Liked: {ingredients_likes}"

    # Add ingredients dislikes to the question
    ingredients_dislikes = ', '.join(persona['ingredients']['dislikes']) if persona['ingredients']['dislikes'] else None
    if ingredients_dislikes:
        question += f"\n - Ingredients Disliked: {ingredients_dislikes}"

    # Add nutrition scores to the question
    nutrition_scores = ', '.join([f"{k}: min. {round(v, 2)}" for k, v in persona['nutrition_scores'].items()]) \
        if persona['nutrition_scores'] else None
    if nutrition_scores:
        question += f"\n - Nutrition Scores: {nutrition_scores}"

    # Add nutrition_limits to the question
    nutrition_limits = ', '.join([f"{k}: {round(v['limits']['lower'], 2)}-{round(v['limits']['upper'], 2)}"
                                  for k, v in persona['relative_nutrients'].items()])
    if nutrition_limits:
        question += f"\n - Nutrition Limits: {nutrition_limits}"

    # Add total_nutrition to the question
    total_nutrition = ', '.join([f"{k}: max. {round(v, 2)}" for k, v in persona['total_nutrients'].items()])
    if total_nutrition:
        question += f"\n - Total Nutrition Limits: {total_nutrition}"

    # If it's a meal plan question, add meal plan restrictions
    if question_type == QuestionType.MEAL_PLAN and meal_plan_restriction:
        meal_plan_info = ''
        for meal_name, guidelines in meal_plan_restriction.items():
            meal_plan_info += f'{meal_name}: '
            meal_plan_info += ', '.join([f"max. {round(value, 2)} {nutrient}"
                                         for nutrient, value in guidelines['guidelines'].items()])
            meal_plan_info += '. '
        question += f"\n - Meal Plan Restriction: {meal_plan_info}"

    return question


def _get_valid_recipes_for_persona(food_on, persona_config, persona: dict, recipes_df: pd.DataFrame,
                                   as_df=False) -> list or pd.DataFrame:
    """
    Filter recipes from the dataframe based on the persona's likes, dislikes, and nutrient limits.

    :param food_on: Pre-processed FoodOn data.
    :param persona_config: The persona configuration.
    :param persona: The persona with likes, dislikes, and nutrient constraints.
    :param recipes_df: The dataframe containing all recipes.
    :param as_df: Whether to return the filtered recipes as a DataFrame (default is False).

    :return: A filtered dict containing recipes that match the persona's preferences.
    """

    # Filter recipes based on relative_nutrients
    for nutrient, limits in persona['relative_nutrients'].items():
        lower_limit = limits['limits']['lower']
        upper_limit = limits['limits']['upper']
        recipes_df = recipes_df[(recipes_df[nutrient] >= lower_limit) & (recipes_df[nutrient] <= upper_limit)]
        if recipes_df.empty:
            if as_df:
                return recipes_df
            else:
                return []

    # Filter recipes based on nutrition_scores (actual score is bigger (==better/healthier) than the limit)
    for score_name, score_limit in persona['nutrition_scores'].items():
        recipes_df = recipes_df[recipes_df[score_name] >= score_limit]
        if recipes_df.empty:
            if as_df:
                return recipes_df
            else:
                return []

    # Filter recipes based on total_nutrients
    for nutrient, total_limit in persona['total_nutrients'].items():
        recipes_df = recipes_df[(recipes_df[nutrient] / 100 * recipes_df['servingSize [g]']) <= total_limit]
        if recipes_df.empty:
            if as_df:
                return recipes_df
            else:
                return []

    # Filter recipes based on liked and disliked tags
    liked_tags = persona['tags']['likes']
    disliked_tags = persona['tags']['dislikes']
    if liked_tags:  # If no liked tags: allow any tag
        recipes_df = recipes_df[recipes_df['tags'].apply(lambda x: any(tag in liked_tags for tag in x))]
        if recipes_df.empty:
            if as_df:
                return recipes_df
            else:
                return []
    recipes_df = recipes_df[~recipes_df['tags'].apply(lambda x: any(tag in disliked_tags for tag in x))]
    if recipes_df.empty:
        if as_df:
            return recipes_df
        else:
            return []

    # Filter recipes based on liked and disliked ingredients
    include_food_on = persona_config['ingredients']['include_food_on']
    include_food_com = persona_config['ingredients']['include_food_com']
    liked_ingr = set(food_on.translate_ingredients(
        ingredients=persona['ingredients']['likes'],
        include_food_on=include_food_on,
        include_food_com=include_food_com))
    disliked_ingr = set(food_on.translate_ingredients(
        ingredients=persona['ingredients']['dislikes'],
        include_food_on=include_food_on,
        include_food_com=include_food_com))

    if liked_ingr:  # If no liked ingredients: allow any ingredient
        recipes_df = recipes_df[recipes_df['ingredient_names'].apply(lambda x: any(ingr in liked_ingr for ingr in x))]
        if recipes_df.empty:
            if as_df:
                return recipes_df
            else:
                return []
    recipes_df = recipes_df[~recipes_df['ingredient_names'].apply(lambda x: any(ingr in disliked_ingr for ingr in x))]

    if as_df:
        return recipes_df
    else:
        return recipes_df['recipe_url'].to_list()


def _generate_valid_meal_plans_for_persona(food_on, persona_config, persona: dict, meal_plan_restriction: dict,
                                           recipes_df: pd.DataFrame) -> dict:
    """
    Generate multiple valid meal plans for a persona based on the given meal plan restrictions and persona preferences.
    Optimizes recipe selection based on proximity to calorie guidelines and nutri_score.

    :param food_on: Pre-processed FoodOn data.
    :param persona_config: The persona configuration.
    :param persona: The persona with likes, dislikes, and nutrient constraints.
    :param meal_plan_restriction: The meal plan restrictions for each meal type.
    :param recipes_df: The dataframe containing all recipes.

    :return: A list of dictionaries, each representing a valid meal plan with assigned recipes.
    """
    # Step 1: Get valid recipes for the persona
    valid_recipes = _get_valid_recipes_for_persona(food_on, persona_config, persona, recipes_df, as_df=True)

    # Step 2: Find possible meal plans
    possible_meal_plans = _find_valid_combinations(valid_recipes, meal_plan_restriction)

    return possible_meal_plans


def _find_valid_combinations(recipes, meal_plan_restrictions):
    valid_combinations = {}
    recipes_df = pd.DataFrame.from_dict(recipes)

    # Iterate through meal types in the meal plan restrictions
    for meal_name, guidelines in meal_plan_restrictions.items():
        meal_type = guidelines['meal_type']
        guideline_recipes = recipes_df.copy()

        # Check if the recipe fits the guidelines (guidelines give the total amount, so we need to multiply by serving
        # size)
        for nutrient, max_value in guidelines['guidelines'].items():
            guideline_recipes = guideline_recipes[(guideline_recipes[nutrient] / 100
                                                   * guideline_recipes['servingSize [g]']) <= max_value]

        guideline_recipes = guideline_recipes['recipe_url'].to_list()
        valid_combinations[meal_name] = {'meal_type': meal_type, 'valid_recipes': guideline_recipes}

    return valid_combinations


def _get_total(recipe: pd.Series, nutrient: str) -> float:
    """
    Calculate the total nutrient content for a given nutrient for a recipe.

    :param recipe: A recipe.
    :param nutrient: The nutrient to calculate the total for.

    :return: The total nutrient content for the given nutrient in the recipe.

    """
    return recipe[nutrient] / 100 * recipe['servingSize [g]']


async def reformulate_questions(question_answer_set, llm, path, file_name, question_type, batch_size=1000,
                                storage_frequency=10000):
    """
    Generate reformulated questions using a language model (LLM) with batching, concurrency, and exception handling.

    :param question_answer_set: A list of dictionaries containing questions and answers.
    :param llm: The language model to use for question generation.
    :param path: The path to the data storage directory.
    :param file_name: The name of the file to store the question-answer set.
    :param question_type: The type of question to generate (recipe or meal plan).
    :param batch_size: Number of concurrent requests to be handled at a time.
    :param storage_frequency: Frequency of storing the question-answer set to a file.
    :return: Updates the question_answer_set in place with 'llm_questions' added.
    """

    async def process_single_qa(qa):
        """
        Process a single question-answer pair, generating reformulated questions.

        :param qa: A single question-answer dictionary.
        """
        if len(qa['llm_questions']) == 0:  # If there are no reformulated questions yet
            try:
                persona_str = '\n' + str(qa['persona'])
                meal_plan_str = 'and \n' + str(qa['meal_plan']) if qa.get('meal_plan') else ''
                query = (
                    f'I have the following original question: \n{qa["question"]}\n\n'
                    'Please create 5 refined and diverse versions of the original question:\n'
                    '- One with a formal tone\n'
                    '- One shorter and more concise\n'
                    '- One casual and conversational\n'
                    '- Two more that you think would be helpful\n\n'
                    f'The following requirements must remain intact: {persona_str}{meal_plan_str}.'
                    'Format the answer as a list of questions as strings, e.g. ["Reformulated question 1", "Reformulated question 2", ...].'
                    'Do not include explanations or anything else except the list of questions.'
                )

                system_msg = {
                    'role': 'system',
                    'content': 'You are a tool that reformulates questions to improve the quality of my question-answer-set. '
                               'Your answers are always the reformulated questions given as a list of strings, '
                               'e.g. ["Reformulated question 1", "Reformulated question 2", ...].'
                               'Do not include explanations or anything else except the list of questions.'
                }

                # Perform the async call to the language model
                response = await acompletion(  # Await the async function properly
                    model=llm.model,
                    messages=[system_msg, {'role': 'user', 'content': query}],
                    api_base=llm.additional_kwargs['api_base'],
                    api_key=llm.additional_kwargs['api_key'],
                    timeout=90,  # fail fast instead of hanging on LiteLLM's 600s default
                    num_retries=3,
                )

                # Extract and deserialize the response content
                response_contents = response.choices[0].message['content']
                serialized_response = load_util.serialize(response_contents)

                if serialized_response:
                    serialized_response = [line.replace('"', '').replace("'", "").replace("+", "").replace("*", "")
                                           .strip() for line in serialized_response]
                else:
                    serialized_response = [cleaned_line for line in response_contents.splitlines()
                                           if (cleaned_line := line.replace('"', '').replace("'", "").replace("+", "")
                                               .replace("*", "").strip()).endswith('?')]

                # Update the question-answer set with reformulated questions
                if isinstance(serialized_response, list) or isinstance(serialized_response, set):
                    qa['llm_questions'] = [{'question': llm_question} for llm_question in serialized_response]

            except OpenAIError as e:
                raise ValueError(f"Error connecting to the LLM: {str(e)}")
            except Exception as e:
                print(f"Error processing question: {qa['question']}. Exception: {str(e)}")

    async def process_batch(batch):
        """Process a batch of question-answer sets concurrently."""
        tasks = [process_single_qa(qa) for qa in batch]
        await asyncio.gather(*tasks)

    # Iterate over the question_answer_set in batches
    for i in tqdm(range(0, len(question_answer_set), batch_size), desc='Generating LLM Questions using batches'):
        batch = question_answer_set[i:i + batch_size]
        await process_batch(batch)  # Process each batch concurrently

        if (i * batch_size) % storage_frequency == 0:
            _save(path, file_name, 'llm_qa', question_answer_set, question_type)


async def rate_reformulated_questions(question_answer_set, llm, path, file_name, question_type, batch_size=1000,
                                      storage_frequency=10000):
    """
    Rate reformulated questions by comparing them to the original question and ensuring all requirements are met.

    :param question_answer_set: A list of dictionaries containing the original and reformulated questions.
    :param llm: The language model to use for evaluation.
    :param path: The path to the data storage directory.
    :param file_name: The name of the file to store the question-answer set.
    :param question_type: The type of question to generate (recipe or meal plan).
    :param batch_size: Number of concurrent requests to be handled at a time.
    :param storage_frequency: Frequency of storing the question-answer set to a file.
    :return: Updates the question_answer_set with a 'rating' field for each reformulated question.
    """

    async def rate_single_question(qa):
        """
        Rate the reformulated questions of a single question-answer pair.

        :param qa: A single question-answer dictionary.
        """
        if qa['llm_questions'] and not 'score' in qa['llm_questions'][0]:
            # If there are no scores (exemplary see if score 0 is present) in the reformulated questions yet
            try:
                original_question = qa['question']
                persona_str = '\n' + str(qa['persona'])
                meal_plan_str = 'and \n' + str(qa['meal_plan']) if qa.get('meal_plan') else ''

                if 'llm_questions' in qa:
                    reformulated_questions = [question['question'] for question in qa['llm_questions']]

                    if reformulated_questions:  # If there are reformulated questions
                        # Prepare the query for the LLM to evaluate each reformulated question
                        query = (
                            f"Original Question: {original_question}\n\n"
                            f"Persona: {persona_str}\n"
                            f"Meal Plan: {meal_plan_str}\n" if meal_plan_str else ""
                                                                                  "\nReformulated questions:\n   - " + "\n   - ".join(
                                reformulated_questions) + "\n\n"
                                                          "Rate each reformulated question based on how well it represents the original question and "
                                                          "maintains the content of the persona and the (if present) meal plan."
                                                          "Provide a score between 1 (poor) and 5 (excellent) for each reformulated question."
                                                          "Return the results as a list of integer scores, e.g. [5, 4, 3]."
                        )

                        system_msg = {
                            'role': 'system',
                            'content': 'You are an evaluation tool that rates reformulated questions based on how well they'
                                       ' maintain the original question’s intent, persona, and meal plan. Your answers must'
                                       ' be a list of scores as integers between 1 and 5, e.g. [5, 4, 3].'
                        }

                        # Perform the async call to the language model
                        response = await acompletion(
                            model=llm.model,
                            messages=[system_msg, {'role': 'user', 'content': query}],
                            api_base=llm.additional_kwargs['api_base'],
                            api_key=llm.additional_kwargs['api_key'],
                            timeout=90,  # fail fast instead of hanging on LiteLLM's 600s default
                            num_retries=3,
                        )

                        # Extract and deserialize the response content (rating for each reformulated question)
                        response_contents = response.choices[0].message['content']
                        scores = load_util.serialize(response_contents)

                        # Update the question-answer set with ratings
                        if isinstance(scores, list) or isinstance(scores, set):
                            if len(reformulated_questions) == len(scores):
                                for i, score in enumerate(scores):
                                    qa['llm_questions'][i]['score'] = score

            except OpenAIError as e:
                raise ValueError(f"Error connecting to the LLM: {str(e)}")
            except Exception as e:
                print(f"Error rating question: {qa['question']}. Exception: {str(e)}")

    async def process_batch(batch):
        """Process a batch of question-answer sets concurrently."""
        tasks = [rate_single_question(qa) for qa in batch]
        await asyncio.gather(*tasks)

    # Start the rating process for all reformulated questions
    for i in tqdm(range(0, len(question_answer_set), batch_size), desc='Rating LLM Questions using batches'):
        batch = question_answer_set[i:i + batch_size]
        await process_batch(batch)  # Process each batch concurrently

        if (i * batch_size) % storage_frequency == 0:
            _save(path, file_name, 'scored_qa', question_answer_set, question_type)


def remove_duplicates(question_answer_set):
    """
    Remove duplicate questions from the question-answer set. Solely dependent on persona and meal plan requirements.

    :param question_answer_set: A list of dictionaries containing questions and answers.
    :return: A list of dictionaries containing unique questions and answers.
    """
    seen = set()  # To track unique ('persona', 'meal_plan') pairs
    result = []  # This will store the unique entries

    for entry in question_answer_set:
        # Get 'persona' and 'meal_plan', default to empty dicts if not found
        persona = entry.get('persona', {})
        meal_plan = entry.get('meal_plan', {})

        # Serialize the dicts to JSON strings
        # 'sort_keys=True' ensures consistent key order for accurate comparison
        try:
            persona_str = json.dumps(persona, sort_keys=True)
            meal_plan_str = json.dumps(meal_plan, sort_keys=True)
        except (TypeError, ValueError) as e:
            raise ValueError(f"Error serializing persona or meal plan: {str(e)}")

        identifier = (persona_str, meal_plan_str)

        # Check and update the set in a single step
        if identifier not in seen:
            seen.add(identifier)
            result.append(entry)  # Avoid multiple lookups on the list's append method

    print(f"Removed {len(question_answer_set) - len(result)} duplicate questions.")

    return result


def load_qa_set(data_path, sub_dir_name=None, include_full_recipes=True, do_remove_duplicates=True,
                discard_empty_qas=False, recipes_df=None, disable_meal_plan_set=False):
    """
    Load the question-answer set from a file and also possibly populates it with recipe.

    :param data_path: The path to the data storage directory.
    :param sub_dir_name: The name of the sub-directory (under data/qa) to store the question-answer set. If None, it
    will be stored directly in data/qa.
    :param include_full_recipes: Whether to include full recipes in the question-answer set (default is True).
    :param do_remove_duplicates: Whether to remove duplicate questions.
    :param discard_empty_qas: If True, empty question-answer pairs will be discarded.
    :param recipes_df: The recipes dataframe to use for populating the question-answer set if include_full_recipes is
    True. If None, it will be loaded from the data directory if necessary.
    :param disable_meal_plan_set: If True, the meal plan question-answer set will not be processed.
    :return: The loaded question-answer set.
    """
    qa_path = Path(data_path, SUB_DIRECTORY_NAME, sub_dir_name if sub_dir_name else '')

    try:
        loaded_data = load_util.load_data(qa_path)

        if loaded_data and include_full_recipes:
            print("Loading full recipes for question-answer set(s).")

            if 'meal_plan_question_answer_set' in loaded_data and disable_meal_plan_set:
                print("Meal plan question-answer set is disabled. Removing Meal plan question-answer set.")
                del loaded_data['meal_plan_question_answer_set']

            if recipes_df is None:
                recipes_df = load_util.get_current_hummus(data_path)['recipes_df']

            if 'meal_plan_question_answer_set' in loaded_data and not disable_meal_plan_set:
                qa_set = loaded_data['meal_plan_question_answer_set']
                _ = [answer.update(
                    {'valid_recipes': recipes_df[recipes_df['recipe_url'].isin(answer['valid_recipes'])]})
                    for qa in qa_set for _, answer in qa['answer'].items()]

            if 'recipe_question_answer_set' in loaded_data:
                qa_set = loaded_data['recipe_question_answer_set']
                _ = [qa.update({'answer': recipes_df[recipes_df['recipe_url'].isin(qa['answer'])]}) for qa in qa_set]
    except Exception as e:
        raise ValueError(f"Error loading question-answer set: {str(e)}")

    if discard_empty_qas:
        if 'meal_plan_question_answer_set' in loaded_data and not disable_meal_plan_set:
            print("Discarding empty questions in meal plan question-answer set.")
            loaded_data['meal_plan_question_answer_set'] = [qa for qa in loaded_data['meal_plan_question_answer_set'] if
                                                            not qa['answer'].empty]
            # Count empty questions
            empty_questions = sum(1 for qa in loaded_data['meal_plan_question_answer_set'] if qa['answer'].empty)
            print(f"Removed {empty_questions} empty questions in meal plan question-answer set.")

        if 'recipe_question_answer_set' in loaded_data:
            print("Discarding empty questions in recipe question-answer set.")
            loaded_data['recipe_question_answer_set'] = [qa for qa in loaded_data['recipe_question_answer_set'] if
                                                         not qa['answer'].empty]

            # Count empty questions
            empty_questions = sum(1 for qa in loaded_data['recipe_question_answer_set'] if qa['answer'].empty)
            print(f"Removed {empty_questions} empty questions in recipe question-answer set.")

    if do_remove_duplicates:
        if 'meal_plan_question_answer_set' in loaded_data and not disable_meal_plan_set:
            print("Removing duplicate questions in meal plan question-answer set.")
            loaded_data['meal_plan_question_answer_set'] = remove_duplicates(
                loaded_data['meal_plan_question_answer_set'])
        if 'recipe_question_answer_set' in loaded_data:
            print("Removing duplicate questions in recipe question-answer set.")
            loaded_data['recipe_question_answer_set'] = remove_duplicates(loaded_data['recipe_question_answer_set'])

    # Print total number of questions in each set
    if 'meal_plan_question_answer_set' in loaded_data and not disable_meal_plan_set:
        print(f"Total meal plan questions: {len(loaded_data['meal_plan_question_answer_set'])}")

    if 'recipe_question_answer_set' in loaded_data:
        print(f"Total recipe questions: {len(loaded_data['recipe_question_answer_set'])}")

    # Print total number of unique recipes in each set
    if 'meal_plan_question_answer_set' in loaded_data and not disable_meal_plan_set:
        unique_recipes = set()
        for qa in loaded_data['meal_plan_question_answer_set']:
            for answer in qa['answer'].values():
                unique_recipes.update(answer['valid_recipes']['recipe_id'].to_list())
        print(f"Total unique recipes in meal plan question-answer set: {len(unique_recipes)}")
        del unique_recipes  # Free memory

    if 'recipe_question_answer_set' in loaded_data:
        unique_recipes = set()
        for qa in loaded_data['recipe_question_answer_set']:
            unique_recipes.update(qa['answer']['recipe_id'].to_list())
        print(f"Total unique recipes in recipe question-answer set: {len(unique_recipes)}")
        del unique_recipes # Free memory

    return loaded_data
