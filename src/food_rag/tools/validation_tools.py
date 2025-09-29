import pandas as pd
from llama_index.core import PromptTemplate
from llama_index.core.llms import LLM
from llama_index.core.workflow import Context
from pydantic import Field
from pydantic.fields import FieldInfo
from typing import List

from src.food_rag.meal_plan.meal_plan_structures import MealPlan, MealPlanQuestions, MealPlanValidation
from src.food_rag.meal_plan.meal_type import MealType
from src.food_rag.query_engines.rag import RecipeRAGQueryEngine
from src.food_rag.tools import ingredient_tools
from src.food_rag.tools.generalization_tools import RecipeQuestionGeneralizationTool
from src.food_rag.tools.recipe_tools import RecipeRecommendationTool
from src.food_rag.tools.tool_base import RecipeToolBase
from src.food_rag.utils.recipe_context import RecipeState


class RecipeValidationTool(RecipeToolBase):
    """
    A tool that validates a given recipe by checking its nutritional content with the query.
    """

    TOOL_NAME = "recipe_validation_tool"

    TEMPLATE = PromptTemplate(
        "You are a validation tool. Here is the recipe to validate:\n"
        "{recipe_data}\n\n"
        "Here is the original question:\n"
        "{question}\n\n"
        "Determine if the recipe is valid based on the question. An empty recipe is always invalid.\n"
        "If valid, respond 'VALID' and briefly explain why; if invalid, respond 'INVALID' "
        "and optionally suggest how to fix it.\n\n"
        "Validation Result:"
    )

    FOOD_ON_TEMPLATE = PromptTemplate(
        "You are a validation tool. Here is the recipe to validate:\n"
        "{recipe_data}\n\n"
        "Here are additional optional information about the ingredients and their hierarchy and food classes:\n"
        "{ingredient_hierarchy}\n\n"
        "Here is the original question:\n"
        "{question}\n\n"
        "Determine if the recipe is valid based on the question. An empty recipe is always invalid.\n"
        "If valid, respond 'VALID' and briefly explain why; if invalid, respond 'INVALID' "
        "and optionally suggest how to fix it.\n\n"
        "Validation Result:"
    )

    def __init__(self, llm: LLM, recipe_df: pd.DataFrame, template: PromptTemplate = None, use_food_on: bool = False,
                 ingredient_tool: ingredient_tools.IngredientHierarchyTool = None, verbose: bool = True):
        """
        Initialize the validation tool.

        :param llm: The language model instance.
        :param recipe_df: The Pandas DataFrame containing known recipes.
        :param template: The prompt template for generating the validation result.
                         If None, the default (TEMPLATE) is used.
        :param use_food_on: If True, includes fitting food_on hierarchies and classes to validate the recipe.
        :param ingredient_tool: The ingredient tool to use for food_on validation if use_food_on is True.
        :param verbose: If True, prints the validation result.
        """
        super().__init__(fn=self.validate_recipe, tool_name=self.TOOL_NAME, verbose=verbose)
        self.llm = llm
        self.recipe_df = recipe_df
        self.use_food_on = use_food_on
        self.template = template or (self.TEMPLATE if not use_food_on else self.FOOD_ON_TEMPLATE)

        if use_food_on:
            if ingredient_tool is None:
                raise ValueError("If 'use_food_on' is True, 'ingredient_tool' must be provided.")
            self.ingredient_tool = ingredient_tool

    def validate_recipe(self, ctx: Context, original_question: str = Field(description="The original question."),
                        recipe_title: str = Field(
                            description="The recipe title which shall be validated.")) -> str:
        """Validate the given recipe by checking whether the original question matches to the recipe."""
        if isinstance(original_question, FieldInfo) or isinstance(recipe_title, FieldInfo):
            raise ValueError("The 'original_question' and 'recipe_title' parameters must be provided as strings.")

        current_state = RecipeState.from_context(ctx)

        # Filter DataFrame for relevant entries by recipe title
        recipe = self.recipe_df[self.recipe_df["title"] == recipe_title]

        # Convert relevant DataFrame entries to text/JSON (or summarize them)
        if not recipe.empty:
            recipe_str = recipe.to_json(orient="records")
        else:
            recipe_str = "No matching recipe in the database."

        # Format the prompt
        if self.use_food_on:
            ingredients = [item.strip() for item
                           in ', '.join(str(x) for x in recipe["ingredients"].dropna()).split(',')
                           if item.strip()]  # filter out empty strings
            ingredient_hierarchy = self.ingredient_tool.get_ingredient_hierarchy(ingredients)
            prompt_text = self.template.format(
                recipe_data=recipe_str,
                ingredient_hierarchy=ingredient_hierarchy,
                question=original_question,
            )
        else:
            prompt_text = self.template.format(
                recipe_data=recipe_str,
                question=original_question,
            )

        # Call the LLM to produce a validation result
        llm_response = self.llm.complete(prompt_text)

        # Update the llm usage count
        current_state.increment_llm_counter(parsing_inc=0, sequential_inc=1, total_inc=1)

        if self.verbose:
            print(f"Validation result:\n''{llm_response.text}''\n-----------\n")

        # Update history
        current_state.update_history_item(f"Validated the recipe '{recipe_title}' on the question '{original_question}'"
                                          f" with result:\n"
                                          f"'{MealPlanValidation.shorten_validation_text(llm_response.text)}'")
        current_state.save_to_context()

        return llm_response.text


class MealPlanValidationTool(RecipeToolBase):
    """
    A tool that validates a given meal plan by checking each recipe's suitability
    against the original question. A meal plan is assumed to be a Dict[str, str]
    where each key is a meal type (e.g., 'breakfast') and each value is the recipe title.
    """

    TOOL_NAME = "meal_plan_validation_tool"

    MEAL_PLAN_TEMPLATE = PromptTemplate(
        "You are a meal plan validation tool. Here is the meal plan:\n"
        "{meal_plan_data}\n\n"
        "Here is the question:\n"
        "{question}\n\n"
        "- Determine if this whole meal plan is valid based on the question.\n"
        "- Whitelist requirements (things the user likes or requests) must be present for at least one meal type in the meal plan.\n"
        "- Blacklist requirements (things the user dislikes or wants to avoid) must be absent in all meal types in the meal plan.\n"
        "- Take also into account whether the meal plan makes sense, e.g. avoid repetitions of the same recipes.\n"
        "- An empty meal plan is always invalid.\n"
        "- If valid, respond 'VALID' and explain why.\n"
        "- If invalid, respond 'INVALID', explain why, and suggest how to fix it.\n\n"
        "Validation Result:"
    )

    RECIPE_TEMPLATE = PromptTemplate(
        "You are a recipe validation tool. Here is the recipe to validate:\n"
        "{recipe_data}\n\n"
        "Here is the question:\n"
        "{question}\n\n"
        "- Determine if this recipe is valid based on the question.\n"
        "- Whitelist requirements (things the user likes or requests) should be present in the recipe.\n"
        "- If there are multiple whitelist requirements, at least one must be present.\n"
        "- Blacklist requirements (things the user dislikes or wants to avoid) must be absent in the recipe.\n"
        "- An empty recipe is always invalid.\n"
        "- If valid, respond 'VALID' and explain why.\n"
        "- If invalid, respond 'INVALID', explain why, and suggest how to fix it.\n\n"
        "Validation Result:"
    )

    FOOD_ON_MEAL_PLAN_TEMPLATE = PromptTemplate(
        "You are a meal plan validation tool. Here is the meal plan:\n"
        "{meal_plan_data}\n\n"
        "Here are additional optional information about the ingredients and their hierarchy and food classes:\n"
        "{ingredient_hierarchy}\n\n"
        "Here is the question:\n"
        "{question}\n\n"
        "- Determine if this whole meal plan is valid based on the question.\n"
        "- Whitelist requirements (things the user likes or requests) must be present for at least one meal type in the meal plan.\n"
        "- Blacklist requirements (things the user dislikes or wants to avoid) must be absent in all meal types in the meal plan.\n"
        "- Take also into account whether the meal plan makes sense, e.g. avoid repetitions of the same recipes.\n"
        "- An empty meal plan is always invalid.\n"
        "- If valid, respond 'VALID' and explain why.\n"
        "- If invalid, respond 'INVALID', explain why, and suggest how to fix it.\n\n"
        "Validation Result:"
    )

    FOOD_ON_RECIPE_TEMPLATE = PromptTemplate(
        "You are a recipe validation tool. Here is the recipe to validate:\n"
        "{recipe_data}\n\n"
        "Here are additional optional information about the ingredients and their hierarchy and food classes:\n"
        "{ingredient_hierarchy}\n\n"
        "Here is the question:\n"
        "{question}\n\n"
        "- Determine if this recipe is valid based on the question.\n"
        "- Whitelist requirements (things the user likes or requests) should be present in the recipe.\n"
        "- If there are multiple whitelist requirements, at least one must be present.\n"
        "- Blacklist requirements (things the user dislikes or wants to avoid) must be absent in the recipe.\n"
        "- An empty recipe is always invalid.\n"
        "- If valid, respond 'VALID' and explain why.\n"
        "- If invalid, respond 'INVALID', explain why, and suggest how to fix it.\n\n"
        "Validation Result:"
    )

    def __init__(self, llm: LLM, recipe_df: pd.DataFrame, meal_plan_template: PromptTemplate = None,
                 recipe_template: PromptTemplate = None, use_food_on: bool = False,
                 ingredient_tool: ingredient_tools.IngredientHierarchyTool = None, verbose: bool = True):
        """
        Initialize the meal plan validation tool.

        :param llm: The language model instance.
        :param recipe_df: The Pandas DataFrame containing known recipes.
        :param meal_plan_template: The prompt template for generating the overall meal plan validation result. If None,
        the default (MEAL_PLAN_TEMPLATE) is used.
        :param recipe_template: The prompt template for generating the individual recipe validation result. If None,
        the default (RECIPE_TEMPLATE) is used.
        :param use_food_on: If True, includes fitting food_on hierarchies and classes to validate the recipe.
        :param ingredient_tool: The ingredient tool to use for food_on validation if use_food_on is True.
        :param verbose: If True, prints the validation result(s).
        """
        super().__init__(fn=self.validate_meal_plan, tool_name=self.TOOL_NAME, verbose=verbose)
        self.llm = llm
        self.recipe_df = recipe_df
        self.use_food_on = use_food_on
        self.meal_plan_template = meal_plan_template or (self.MEAL_PLAN_TEMPLATE if not use_food_on
                                                         else self.FOOD_ON_MEAL_PLAN_TEMPLATE)
        self.recipe_template = recipe_template or (self.RECIPE_TEMPLATE if not use_food_on
                                                   else self.FOOD_ON_RECIPE_TEMPLATE)

        if use_food_on:
            if ingredient_tool is None:
                raise ValueError("If 'use_food_on' is True, 'ingredient_tool' must be provided.")
            self.ingredient_tool = ingredient_tool

    def validate_meal_plan(self, ctx: Context) -> str:
        """
        Validate the given meal plan by checking if each recipe aligns with the original question.
        """
        return self.validate_meal_plan_parts(ctx=ctx)  # Validate the entire meal plan

    def validate_meal_plan_parts(self, ctx: Context, meal_types: List[MealType] = None) -> str:
        """
        Validate the given meal plan by checking if each recipe aligns with the original question.

        :param ctx: The context object containing the state.
        :param meal_types: A list of enabled meal types to validate. Validates all enabled meal types which are in the
         meal recommendation in the context. If None, all in the context present meal types are validated. This is
          useful for a partial recommendation.
        """
        current_state = RecipeState.from_context(ctx)

        original_question = current_state.get_original_user_question()
        meal_plan_questions: MealPlanQuestions = current_state.get_meal_plan_questions()
        meal_plan: MealPlan = current_state.get_meal_plan_recommendation()

        # Use partial validation if meal_types is provided
        if meal_types is not None:
            meal_plan = MealPlan.from_dict({meal_type: meal for meal_type, meal in meal_plan.items() if meal_type
                                            in meal_types})

        # Overall validation of the meal plan
        overall_validation = self._overall_validation(current_state, original_question, meal_plan)

        # Validate each recipe in the meal plan and update the meal_plan_validation dict
        meal_plan_validation = self._validation(current_state, meal_plan_questions, meal_plan)

        # Store the validation results
        current_state.set_overall_meal_plan_validation(overall_validation)
        current_state.update_meal_plan_validation(meal_plan_validation)

        # Update history
        history_str = (f"Validated the meal_plan with the following results:\n"
                       f"Overall validation: '{MealPlanValidation.shorten_validation_text(overall_validation)}'\n\n"
                       f"Validation for each recipe recommendation separately: {meal_plan_validation.get_short_str()}")
        current_state.update_history_item(history_str)

        current_state.save_to_context()

        return history_str

    def _overall_validation(self, current_state: RecipeState, original_question, meal_plan) -> str:
        # Build a prompt for the meal plan
        if self.use_food_on:
            ingredients = [recipe.get("ingredients", "") for recipe in meal_plan.values()]
            ingredients = [item.strip() for item in ', '.join(str(x) for x in ingredients).split(',')
                           if item.strip()]  # filter out empty strings
            ingredient_hierarchy = self.ingredient_tool.get_ingredient_hierarchy(list(set(ingredients)))
            prompt_text = self.meal_plan_template.format(
                meal_plan_data=meal_plan,
                ingredient_hierarchy=ingredient_hierarchy,
                question=original_question,
            )
        else:
            prompt_text = self.meal_plan_template.format(
                meal_plan_data=meal_plan,
                question=original_question,
            )

        if self.verbose:
            print(f"Meal Plan Validation Input Prompt:\n{prompt_text}")

        # Run the LLM to validate this single meal
        llm_response = self.llm.complete(prompt_text)

        # Update the llm usage count
        current_state.increment_llm_counter(parsing_inc=0, sequential_inc=1, total_inc=1)

        if self.verbose:
            print(f"Meal Plan Validation Results:\n{llm_response.text}")

        return llm_response.text

    def _validation(self, current_state: RecipeState, meal_plan_question, meal_plan) -> MealPlanValidation:
        meal_plan_validation = MealPlanValidation()
        for meal_type, recipe_data in meal_plan.items():
            # Build a prompt for the recipe
            if self.use_food_on:
                ingredients = recipe_data.get("ingredients", "")
                ingredients = [item.strip() for item in ingredients.split(',') if item.strip()]
                ingredient_hierarchy = self.ingredient_tool.get_ingredient_hierarchy(ingredients)
                prompt_text = self.recipe_template.format(
                    recipe_data=recipe_data,
                    ingredient_hierarchy=ingredient_hierarchy,
                    question=meal_plan_question[meal_type],
                )
            else:
                prompt_text = self.recipe_template.format(
                    recipe_data=recipe_data,
                    question=meal_plan_question[meal_type],
                )

            if self.verbose:
                print(f"Recipe Validation Input Prompt:\n{prompt_text}")

            # Run the LLM to validate this single meal
            llm_response = self.llm.complete(prompt_text)

            if self.verbose:
                print(f"Recipe Validation Results:\n{llm_response.text}")

            # Store the validation result
            meal_plan_validation[meal_type] = llm_response.text

        # Update the llm usage count by taking parallelizable calls into account
        parallelizable_counts = len(meal_plan.items())
        current_state.increment_llm_counter(parsing_inc=0, sequential_inc=1, total_inc=parallelizable_counts)

        return meal_plan_validation


class CorrectionTool(RecipeToolBase):
    """
    A tool that corrects a given recipe by checking its nutritional content with the query.
    """

    TOOL_NAME = "recipe_correction_tool"

    def __init__(self, llm: LLM, recipe_df: pd.DataFrame, recipe_query_engine: RecipeRAGQueryEngine,
                 use_food_on: bool = False, ingredient_tool: ingredient_tools.IngredientHierarchyTool = None,
                 verbose: bool = True):
        """
        Initialize the correction tool.

        :param llm: The language model instance.
        :param recipe_df: The Pandas DataFrame containing known recipes.
        :param use_food_on: If True, includes fitting food_on hierarchies and classes to validate the recipe.
        :param ingredient_tool: The ingredient tool to use for food_on validation if use_food_on is True.
        :param verbose: If True, prints the correction result.
        """
        super().__init__(fn=self.correct_recipes, tool_name=self.TOOL_NAME, verbose=verbose)
        self.llm = llm
        self.recipe_df = recipe_df
        self.meal_plan_validation_tool = MealPlanValidationTool(llm=llm, recipe_df=recipe_df, verbose=verbose,
                                                                use_food_on=use_food_on,
                                                                ingredient_tool=ingredient_tool)
        self.recipe_validation_tool = RecipeValidationTool(llm=llm, recipe_df=recipe_df, verbose=verbose)
        self.generalization_tool = RecipeQuestionGeneralizationTool(llm=llm, verbose=verbose)
        self.recipe_recommendation_tool = RecipeRecommendationTool(recipe_query_engine=recipe_query_engine,
                                                                   verbose=verbose)

    def correct_recipes(self, ctx: Context) -> str:
        """
        Fix invalid recipes by generalizing their questions up to 5 levels.
        If they stay invalid, restore the originals.
        """
        max_gen_level = self.generalization_tool.MAX_GENERALIZATION_LEVEL

        # 1. Load state
        current_state = RecipeState.from_context(ctx)

        # 2. Stash originals and init current plans
        original_meal_plan_question: MealPlanQuestions = current_state.get_meal_plan_questions()
        original_meal_plan_recommendation: MealPlan = current_state.get_meal_plan_recommendation()

        # 3. Find invalid questions
        self.meal_plan_validation_tool.validate_meal_plan_parts(ctx=ctx,
                                                                meal_types=list(original_meal_plan_question.keys()))
        current_state.reload_from_context()
        meal_plan_validation: MealPlanValidation = current_state.get_meal_plan_validation()
        original_invalid_questions: MealPlanQuestions = self._get_invalid_questions(original_meal_plan_question,
                                                                           meal_plan_validation)
        invalid_questions = original_invalid_questions.deepcopy()

        # 4. Generalize and re-validate up to MAX_GENERALIZATION_LEVEL
        for level in range(1, max_gen_level + 1):
            if not invalid_questions:
                break

            # generalize each invalid question
            new_questions: MealPlanQuestions = self._generalize_questions(ctx, level, invalid_questions)
            current_state.update_meal_plan_questions(new_questions)

            # re-recommend recipes
            new_recommendations: MealPlan = self._recommend_recipes(ctx=ctx, meal_plan_questions=new_questions)
            current_state.update_meal_plan_recommendation(new_recommendations)

            # Update the context
            current_state.save_to_context()

            # re-validate
            self.meal_plan_validation_tool.validate_meal_plan_parts(ctx=ctx, meal_types=list(invalid_questions.keys()))

            # reload the state to check for remaining invalid questions
            current_state.reload_from_context()
            meal_plan_question = current_state.get_meal_plan_questions()
            meal_plan_validation = current_state.get_meal_plan_validation()
            invalid_questions: MealPlanQuestions = self._get_invalid_questions(meal_plan_question, meal_plan_validation)

        # 5. Restore originals for invalid recommendations and update the history accordingly
        meal_plan = current_state.get_meal_plan_recommendation()
        updated_meal_types = [meal_type for meal_type in original_invalid_questions.keys()
                              if meal_type not in invalid_questions]
        new_meals = {meal_type: meal_plan[meal_type] for meal_type in updated_meal_types}
        if invalid_questions:
            self._restore_originals(invalid_questions, current_state,
                                    original_meal_plan_question, original_meal_plan_recommendation)

            # Update history
            if new_meals:
                history_str = (f"Tried to correct the meal plan but failed to find better recipes for"
                               f" the meal types: {invalid_questions.keys()}.\n"
                               f"Successful changes made to the following meal types with the updated "
                               f"recommendation:\n"
                               f"{MealPlan.from_dict(new_meals).get_short_str()}")
                current_state.update_history_item(history_str)
            else:
                history_str = (f"Tried to correct the meal plan but failed to find better recipes for"
                               f" the meal types: {invalid_questions.keys()}.")
                current_state.update_history_item(history_str)
        else:
            # Update history
            if new_meals:
                history_str = (f"Corrected and improved the meal plan by generalizing invalid questions"
                               f" and re-recommending recipes. Successful changes made to the following"
                               f" meal types with the updated recommendation:\n"
                               f"{MealPlan.from_dict(new_meals).get_short_str()}")
                current_state.update_history_item(history_str)
            else:
                history_str = "Validated the meal plan successfully."
                current_state.update_history_item(history_str)

        # Update the context
        current_state.save_to_context()

        return history_str

    def _get_invalid_questions(self, meal_plan_question: MealPlanQuestions,
                               meal_plan_validation: MealPlanValidation) -> MealPlanQuestions:
        invalid_questions = {
            recipe_type.value: meal_plan_question[recipe_type] for recipe_type, status in meal_plan_validation.items()
            if "INVALID" in status
        }
        return MealPlanQuestions.from_dict(invalid_questions)

    def _generalize_questions(self, ctx, generalization_level, invalid_questions: MealPlanQuestions) \
            -> MealPlanQuestions:
        new_questions = MealPlanQuestions()
        for meal_type, old_question in invalid_questions.items():
            new_question = self.generalization_tool.generalize_question(
                ctx=ctx,
                question=old_question,
                generalization_level=generalization_level
            )
            new_questions[meal_type] = new_question
        return new_questions

    def _recommend_recipes(self, ctx: Context, meal_plan_questions: MealPlanQuestions) -> MealPlan:
        meal_plan_recommendation = MealPlan()
        for meal_type, meal_plan_question in meal_plan_questions.items():
            meal_plan_recommendation[meal_type] = (
                self.recipe_recommendation_tool.recommend_recipe(ctx=ctx, question=meal_plan_question))
        return meal_plan_recommendation

    def _restore_originals(self, invalid_questions, current_state, original_meal_plan_question,
                           original_meal_plan_recommendation):
        for meal_type in invalid_questions.keys():
            current_state.update_meal_plan_questions_item(meal_type, original_meal_plan_question[meal_type])
            current_state.update_meal_plan_recommendation_item(meal_type, original_meal_plan_recommendation[meal_type])
