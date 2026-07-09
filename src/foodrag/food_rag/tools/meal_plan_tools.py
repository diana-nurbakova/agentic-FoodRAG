import pandas as pd
from datetime import datetime
from llama_index.core import PromptTemplate
from llama_index.core.base.response.schema import RESPONSE_TYPE, Response
from llama_index.core.llms import LLM
from llama_index.core.workflow import Context
from pydantic import Field
from pydantic.fields import FieldInfo

from foodrag.food_rag.meal_plan.meal_plan_structures import MealPlanQuestions, MealPlan
from foodrag.food_rag.meal_plan.meal_type import MealType
from foodrag.food_rag.query_engines.rag import RecipeRAGQueryEngine, LLM_DF_INPUT_KEY
from foodrag.food_rag.tools.tool_base import RecipeToolBase
from foodrag.food_rag.tools.util_tools import JsonExtractorTool
from foodrag.food_rag.utils.recipe_context import RecipeState
from foodrag.utils import hummus_util

QUERY_AUGMENTATION_TEMPLATE = PromptTemplate(
    'Split the given question into multiple meal-plan questions:\n\n'
    '- Keep all information from the original question in the meal plan.\n'
    '- You can split whitelist requirements (things the user likes or asks for) into multiple meal plan questions.\n'
    '- If mentioned, assign whitelist requirements (things the user likes) to the respective meal type, otherwise you can assign it to one or multiple meal type questions.\n'
    '- Optionally, you can assign your own whitelist requirements to any meal type question to balance the meal plan.\n'
    '- You must keep blacklist requirements (things the user dislikes) in all meal plan questions.\n'
    '- All (blacklist and whitelist) requirements are directly given in the original question.\n'
    '- Each question should be for a single meal type.\n'
    '- Each question should be compact and only contain for the retrieval necessary information.\n'
    '- Preserve all nutritional and dietary requirements.\n'
    '- Use only these meal types: ' + ', '.join([f'"{mt}"' for mt in MealType.get_all_str()]) +
    '.\n'
    '- Include meal types relevant to the current time: {time}.\n'
    '- Optionally, include meal types relevant to the current season given by the date: {date}.\n'
    '- Return the output as a JSON object with meal types as keys and questions as values. For example:\n'
    '```json\n{"meal type 1": "Question 1", "meal type 2": "Question 2", ...}\n```\n\n'
    'Current date and time: {date}, {time}\n'
    'Original Question: {question}\n\n'
    'Meal Plan Questions:')

# Full-day variant: request a meal-plan question for EVERY meal type (breakfast,
# lunch, dinner, snack) irrespective of the current time. Used when the tool is
# constructed with ``full_day=True`` so the pipeline produces a complete 4-slot plan
# instead of only the meals "relevant to the current time".
FULL_DAY_QUERY_AUGMENTATION_TEMPLATE = PromptTemplate(
    'Split the given question into multiple meal-plan questions covering a FULL DAY:\n\n'
    '- Generate a meal-plan question for EVERY meal type: '
    + ', '.join([f'"{mt}"' for mt in MealType.get_all_str()]) +
    ' — exactly one question per meal type, one per meal type.\n'
    '- Keep all information from the original question in the meal plan.\n'
    '- Distribute the user\'s whitelist requirements (things the user likes or asks for) '
    'across the meal-type questions.\n'
    '- If mentioned, assign whitelist requirements to the respective meal type, otherwise '
    'distribute them across one or multiple meal type questions to balance the meal plan.\n'
    '- Optionally, you can assign your own whitelist requirements to any meal type question '
    'to balance the meal plan.\n'
    '- You MUST keep all blacklist requirements (things the user dislikes) in EVERY meal-plan question.\n'
    '- All (blacklist and whitelist) requirements are directly given in the original question.\n'
    '- Each question should be for a single meal type and be compact, only containing for the '
    'retrieval necessary information.\n'
    '- Preserve all nutritional and dietary requirements.\n'
    '- Use only these meal types: ' + ', '.join([f'"{mt}"' for mt in MealType.get_all_str()]) + '.\n'
    '- IGNORE the current time: include ALL FOUR meal types regardless of the time of day.\n'
    '- Optionally, tailor the meal types to the season given by the date: {date}.\n'
    '- Return the output as a JSON object with meal types as keys and questions as values. For example:\n'
    '```json\n{"breakfast": "Question 1", "lunch": "Question 2", "dinner": "Question 3", '
    '"snack": "Question 4"}\n```\n\n'
    'Current date and time: {date}, {time} (ignore the time; still include all four meal types)\n'
    'Original Question: {question}\n\n'
    'Meal Plan Questions:')


class MealPlanRecommendationTool(RecipeToolBase):
    """
    A tool that generates meal plans for the remaining day.
    """
    TOOL_NAME = "meal_plan_recommendation_tool"

    TOOL_NAME_RETRIEVE_ONLY = "meal_plan_alternatives_tool"

    def __init__(self, llm: LLM, recipe_query_engine: RecipeRAGQueryEngine, retrieve_only=False,
                 query_augmentation_template: PromptTemplate = None, time: str = None, date: str = None,
                 full_day: bool = False, verbose=True):
        """
        Initialize the MealPlanRecommendationTool.

        :param LLM llm: The language model instance.
        :param RecipeRAGQueryEngine recipe_query_engine: The base recipe question engine instance.
        :param bool retrieve_only: Whether to only retrieve the recipes directly from the by the query engine used
         retriever, or to use the whole query engine. Defaults to False which uses the whole query engine.
        :param PromptTemplate query_augmentation_template: Template for generating meal plans. Defaults to the
        QUERY_AUGMENTATION_TEMPLATE (or FULL_DAY_QUERY_AUGMENTATION_TEMPLATE when ``full_day`` is True).
        :param str time: The time of day for the recommendation (e.g., "morning", "noon", "evening", "12am" , "10:20",
        ...). If None, "now", or "current" it is automatically set to the current time during query execution.
        :param str date: The date for the recommendation (e.g., "today", "tomorrow", "2023-10-01", ...). If None, it is
        automatically set to the current date during query execution.
        :param bool full_day: When True, reformulate the query into a meal-plan question for ALL FOUR meal types
        (breakfast, lunch, dinner, snack) regardless of the current time, producing a complete 4-slot meal plan.
        Defaults to False (time-aware partial-day plan). Ignored when an explicit ``query_augmentation_template`` is
        given.

        :param bool verbose: Whether to print details of the generation process.
        """
        if retrieve_only:
            super().__init__(fn=self.recommend_meal_plan, tool_name=self.TOOL_NAME_RETRIEVE_ONLY, verbose=verbose)
        else:
            super().__init__(fn=self.recommend_meal_plan, tool_name=self.TOOL_NAME, verbose=verbose)
        self.llm = llm
        self.recipe_query_engine = recipe_query_engine
        self.json_extractor_tool = JsonExtractorTool(llm=recipe_query_engine.llm, verbose=verbose)
        self.retrieve_only = retrieve_only
        self.full_day = full_day
        if query_augmentation_template is not None:
            self.query_augmentation_template = query_augmentation_template
        elif full_day:
            self.query_augmentation_template = FULL_DAY_QUERY_AUGMENTATION_TEMPLATE
        else:
            self.query_augmentation_template = QUERY_AUGMENTATION_TEMPLATE
        self.time = time
        self.date = date

    def recommend_meal_plan(self, ctx: Context,
                            question: str = Field(description=f"A food related question as string.")) -> MealPlan:
        """Recommends a meal plan based on the given question."""
        if isinstance(question, FieldInfo):
            raise ValueError("The 'question' parameter must be provided as a string.")

        self.get_meal_plan_question(ctx=ctx, question=question)
        meal_plan = self.get_meal_plan(ctx=ctx)

        return meal_plan

    def get_meal_plan_question(self, ctx: Context,
                               question: str = Field(description=f"A food related question as string.")):
        """
        Reformulates a single recipe question into a meal plan question by taking into account the current time and
         available meal types.

        :param Context ctx: The context object to store the state.
        :param str question: A food related question as string.
        :return: Generated meal plan question as a structured response.
        """
        current_state = RecipeState.from_context(ctx)
        current_state.set_original_user_question(question)

        # Transform the question into meal plan questions
        reformulated_meal_plan_query = self._reformulate_query(current_state=current_state, llm=self.llm,
                                                               question=question, time=self.time, date=self.date,
                                                               query_augmentation_template=self.query_augmentation_template,
                                                               verbose=self.verbose).response
        meal_plan_query = self._parse_meal_plan_query(reformulated_meal_plan_query)
        current_state.set_meal_plan_questions(meal_plan_query)

        # Update history
        current_state.update_history_item(f"Reformulated the question '{question}' into meal plan questions:\n"
                                          f"{meal_plan_query}.")

        current_state.save_to_context()

    def get_meal_plan(self, ctx: Context) -> MealPlan:
        """
        Recommends a new meal plan based on the given question.

        :param Context ctx: The context object to store the state.
        :return: Generated meal plan.
        :rtype: MealPlan
        """
        current_state = RecipeState.from_context(ctx)

        # Fetch the meal plan question from the state
        meal_plan_query: MealPlanQuestions = current_state.get_meal_plan_questions()
        if meal_plan_query is None:
            raise ValueError("No meal plan question found in the state. Please call get_meal_plan_question first.")

        # Use the query engine to retrieve recipes for each meal for the meal plan
        meal_plan = MealPlan()
        for meal_type, meal_query in meal_plan_query.items():
            # Validate type
            if str(meal_type) not in MealType.get_all_str():
                raise ValueError(f"Invalid meal type: {str(meal_type)}. Valid meal types are: {MealType.get_all_str()}")

            # Add the meal type to the query
            query = f"For {str(meal_type)}: {meal_query}"

            if self.retrieve_only:
                recommended_recipe_df = self.recipe_query_engine.retrieve(query)
            else:
                recipe_response = self.recipe_query_engine.query(query)
                retrieved_recipes = recipe_response.metadata.get(LLM_DF_INPUT_KEY, pd.DataFrame())

                # Try to extract the json from the response (output between  ```json and ```).
                try:
                    response_json_str = recipe_response.response.split('```json')[1].split('```')[0].strip()
                except Exception:
                    response_json_str = self.json_extractor_tool.extract_json(ctx=ctx, text=recipe_response.response)

                    if response_json_str == "":
                        # If no JSON was found, use the original response
                        response_json_str = recipe_response.response

                # Try to parse the JSON
                try:
                    response_json = hummus_util.try_parse(response_json_str)

                    if 'title' in response_json and response_json["title"] in retrieved_recipes["title"].values:
                        recommended_recipe_df = retrieved_recipes[retrieved_recipes["title"] == response_json["title"]]
                    else:
                        recommended_recipe_df = retrieved_recipes
                except SyntaxError:
                    recommended_recipe_df = retrieved_recipes

            recommended_recipe_dict = (recommended_recipe_df.head(1).to_dict(orient="records")[0]
                                       if not recommended_recipe_df.empty else {})
            meal_plan[meal_type] = recommended_recipe_dict

        current_state.set_meal_plan_recommendation(meal_plan)

        # Fetches the llm calls done per meal type
        if self.retrieve_only:
            c = self.recipe_query_engine.llm_calls - 1  # -1 because we use the retriever only
        else:
            c = self.recipe_query_engine.llm_calls

        # Update the llm usage count by taking parallelizable calls into account
        parallelizable_counts = len(meal_plan_query.items())
        current_state.increment_llm_counter(parsing_inc=0, sequential_inc=c, total_inc=c * parallelizable_counts)

        # Update history
        current_state.update_history_item(f"Based on those meal plan questions:\n"
                                          f"{meal_plan_query}\n"
                                          f"recommended this meal plan:\n"
                                          f"{meal_plan.get_short_str()}")

        current_state.save_to_context()

        return meal_plan

    @staticmethod
    def _reformulate_query(current_state: RecipeState, llm, question, time, date, query_augmentation_template,
                           verbose) -> RESPONSE_TYPE:
        if time is None or (isinstance(time, str) and time.lower() in ['now', 'current']):
            # fetch the systems time
            time = datetime.now().strftime('%H:%M')

        if date is None or (isinstance(date, str) and date.lower() in ['now', 'today', 'current']):
            # fetch the systems date
            date = datetime.now().strftime('%Y-%m-%d')

        # Persist the RESOLVED date+time (whether pinned or from datetime.now()) into the
        # state so full-pipeline runs can record + reproduce which time — and therefore
        # which meal types — were requested. Saved by the caller via save_to_context().
        current_state.set_recommendation_time(f"{date} {time}")

        # Use the LLM to generate a meal plan
        augmentation_prompt = query_augmentation_template.format(question=question, time=str(time), date=str(date))

        if verbose:
            print(f"Meal Plan augmentation instruction:\n''{augmentation_prompt}''\n--------------\n")

        # Generate the reformulated meal plan question
        reformulated_meal_plan = llm.complete(augmentation_prompt)

        # Update the llm usage count
        current_state.increment_llm_counter(parsing_inc=0, sequential_inc=1, total_inc=1)

        if verbose:
            print(f"Reformulated meal plan question:\n''{reformulated_meal_plan.text}''\n--------------\n")

        return Response(response=reformulated_meal_plan.text, metadata=reformulated_meal_plan.additional_kwargs)

    @staticmethod
    def _parse_meal_plan_query(reformulated_meal_plan_query) -> MealPlanQuestions:
        # Try to extract the json from the generated meal plan context (output between  ```json and ```).
        try:
            meal_plan_json_str = reformulated_meal_plan_query.split('```json')[1].split('```')[0].strip()
        except Exception:
            meal_plan_json_str = reformulated_meal_plan_query

        #  Try to parse the meal plan JSON and retrieve recipes for each meal
        try:
            meal_plan_dict = hummus_util.try_parse(meal_plan_json_str)
        except SyntaxError:
            meal_plan_dict = {}

        # Convert the JSON to a MealPlanQuery object
        meal_plan_query = MealPlanQuestions.from_dict(meal_plan_dict)
        return meal_plan_query


class MealPlanUpdateTool(RecipeToolBase):
    """
    A tool that updates one meal type in the current meal plan with a new recommendation.
    """
    TOOL_NAME = "meal_plan_update_tool"

    def __init__(self, verbose=True):
        super().__init__(fn=self.update_meal_plan, tool_name=self.TOOL_NAME, verbose=verbose)

    def update_meal_plan(self, ctx: Context,
                         meal_type: str = Field(
                             description=f"Meal type to update (must be one of {MealType.get_all_str()})."
                         ),
                         recipe_recommendation: str = Field(
                             description="New recommendation (e.g., updated recipe)."
                         ),
                         ) -> str:
        """
        Updates the existing meal plan for the given meal_type with a new recommendation.
        Returns the updated meal plan.
        """
        current_state = RecipeState.from_context(ctx)

        # Validate the meal type
        if meal_type not in MealType.get_all_str():
            raise ValueError(f"Invalid meal type: {meal_type}. Valid meal types: {MealType.get_all_str()}")

        # Update the meal plan entry
        current_state.update_meal_plan_recommendation_item(MealType(meal_type), recipe_recommendation)
        current_state.save_to_context()

        return "Updated meal plan successfully."
