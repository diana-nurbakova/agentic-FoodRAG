from llama_index.core import PromptTemplate
from llama_index.core.llms import LLM
from llama_index.core.workflow import Context
from typing import List

from foodrag.food_rag.meal_plan.meal_plan_structures import MealPlanValidation, MealPlan, MealPlanSubstitution
from foodrag.food_rag.tools.tool_base import RecipeToolBase
from foodrag.food_rag.tools.util_tools import JsonExtractorTool
from foodrag.food_rag.utils.recipe_context import RecipeState


class SubstitutionTool(RecipeToolBase):
    """
    A tool that uses an LLM to modify recipes based on user-specified objectives
    like making the recipe healthier, cheaper, easier to cook, or adapting to user feedback.
    """

    TOOL_NAME = "substitution_tool"

    SUBSTITUTION_TEMPLATE = PromptTemplate(
        "Modify the following {meal_type} recipe recommendation according to the provided objectives and feedback:\n"
        "- Use substitutions or simplifications only where it fits the objective or provided feedback.\n"
        "- Keep the recipe identity intact, do not totally change it.\n"
        "- Focus only on ingredients and directions. You may adjust quantities, ingredients, or steps.\n"
        "- Also consider previous validation feedback to fix issues.\n"
        "- If the original recipe is already good, just return it like it is.\n"
        "- If the original recipe is empty or not a recipe, return an empty recipe.\n"
        "- Return the modified recipe as a json object, like for example:\n"
        '   ```json\n      {"title": "Recipe Title", "directions": "...", "ingredients": "...", ...}\n   ```\n\n'
        "Objectives:\n{objectives}\n\n"
        "Recipe recommendation:\n{recipe_recommendation}\n\n"
        "Overall meal plan feedback (might also includes irrelevant information for the {meal_type} recommendation):"
        "\n{overall_feedback}\n\n"
        "Direct feedback:\n{recipe_feedback}\n\n"
        "Modified recipe:"
    )

    def __init__(self, llm: LLM, substitution_template: PromptTemplate = None, verbose: bool = True):
        """
        Initialize the substitution tool.

        :param llm: The language model instance.
        :param substitution_template: Optional prompt template.
        :param verbose: If True, print debugging information.
        """
        super().__init__(fn=self.substitute, tool_name=self.TOOL_NAME, verbose=verbose)
        self.llm = llm
        self.substitution_template = substitution_template or self.SUBSTITUTION_TEMPLATE
        self.json_extractor_tool = JsonExtractorTool(llm=llm, verbose=verbose)

    def substitute(self, ctx: Context, objectives: List[str]) -> str:
        """
        Modify the recipe in the current state according to objectives and validation feedback.

        :param ctx: The workflow context.
        :param objectives: List of modification goals (e.g., ['healthier', 'cheaper']).
        """
        current_state = RecipeState.from_context(ctx)
        meal_plan: MealPlan = current_state.get_meal_plan_recommendation()
        overall_feedback: str = self.format_feedback(current_state.get_overall_meal_plan_validation())
        validation_feedback: MealPlanValidation = current_state.get_meal_plan_validation()
        substituted_meal_plan: MealPlanSubstitution = current_state.get_substituted_meal_plan_recommendation()

        for meal_type, recipe in meal_plan.items():
            recipe_feedback = self.format_feedback(validation_feedback.get(meal_type, "No feedback available."))
            recipe_recommendation = meal_plan.get_partial_recommendation_str(
                meal_types=[meal_type], features=['title', 'directions', 'ingredient_quantities'])

            prompt = self.substitution_template.format(
                meal_type=meal_type,
                objectives=", ".join(objectives),
                recipe_recommendation=recipe_recommendation,
                overall_feedback=overall_feedback,
                recipe_feedback=recipe_feedback,
            )

            if self.verbose:
                print(f"Substitution prompt for {meal_type}:\n''{prompt}''\n--------------\n")

            response = self.llm.complete(prompt)

            if self.verbose:
                print(f"Modified recipe for {meal_type}:\n''{response.text}''\n--------------\n")

            # Parse the response to extract the modified recipe
            # Try to extract the json from the response (output between  ```json and ```).
            try:
                response_json_str = response.text.split('```json')[1].split('```')[0].strip()
            except Exception:
                response_json_str = self.json_extractor_tool.extract_json(ctx=ctx, text=response.text)

                if response_json_str == "":
                    # If no JSON was found, use the original response
                    response_json_str = response.text

            substituted_meal_plan[meal_type] = response_json_str

            # Update history (include possible explanation in history)
            current_state.update_history_item(f"Based on the validation feedback and the objectives '{objectives}', "
                                              f"the recipe for {meal_type} was modified:\n"
                                              f"{response_json_str}")

        # Update the llm usage count by taking parallelizable calls into account (excluding the json extractor tool,
        # we cannot make any assumptions there as we don't know when it is used, so we just statically always count it
        # inside the tool.
        parallelizable_counts = len(meal_plan.items())
        current_state.increment_llm_counter(parsing_inc=0, sequential_inc=1, total_inc=parallelizable_counts)

        current_state.set_substituted_meal_plan_recommendation(substituted_meal_plan)
        current_state.save_to_context()

        return str(substituted_meal_plan)

    @staticmethod
    def format_feedback(feedback: str) -> str:
        """
        Format the feedback string for better readability.

        :param feedback: The feedback string to format.
        :return: The formatted feedback string.
        """
        cleaned_string = (feedback.replace("VALID \n\n", "VALID:")
                          .replace("VALID\n\n", "VALID:")
                          .replace("VALID\n", "VALID:")
                          .replace("VALID \n", "VALID:"))
        cleaned_string = "\n".join(line for line in cleaned_string.splitlines() if line.strip())
        return cleaned_string
