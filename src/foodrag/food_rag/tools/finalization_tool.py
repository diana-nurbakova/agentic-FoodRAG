from llama_index.core import PromptTemplate
from llama_index.core.llms import LLM
from llama_index.core.workflow import Context

from foodrag.food_rag.meal_plan.meal_plan_structures import MealPlanSubstitution
from foodrag.food_rag.meal_plan.meal_plan_structures import MealPlan
from foodrag.food_rag.tools.tool_base import RecipeToolBase
from foodrag.food_rag.utils.recipe_context import RecipeState

QUERY_REFINEMENT_TEMPLATE = PromptTemplate(
    "- Reformulate the meal plan recommendation and its explanation\n"
    "- The recommendation must be, content-wise, kept unchanged.\n"
    "- Do not include any new information, just improve the structure and reformulate.\n\n"
    "Meal plan recommendation:\n"
    "{meal_plan}\n\n"
    "Explanation:\n"
    "{explanation}\n\n"
    "Final recommendation:\n"
)

MODE_SIMPLE = "simple"
MODE_LLM_REFINED = "llm_refined"
MODES = [MODE_SIMPLE, MODE_LLM_REFINED]


class FinalizeRecommendationTool(RecipeToolBase):
    """
    A tool that finalizes the meal plan recommendation by looking at the context, collecting the meal plan and
    concatenating it with the explanation.
    """

    TOOL_NAME = "finalize_meal_plan_recommendation_tool"

    def __init__(self, llm: LLM, mode: str = MODE_SIMPLE, use_substitution: bool = False,
                 query_refinement_template: PromptTemplate = None, verbose=True):
        """
        Initialize the finalization tool.
        :param LLM llm: The language model instance.
        :param str mode: The mode of the tool. Defaults to MODE_SIMPLE.
        :param PromptTemplate query_refinement_template: The template for the query refinement. Defaults to
            QUERY_REFINEMENT_TEMPLATE.
        :param bool use_substitution: Whether to use the generated substitution recipe if available. Defaults to False.
        If it is False it only uses the substitution if the whole meal plan is empty.
        :param bool verbose: Whether to print details of the generation process.
        """
        if mode not in MODES:
            raise ValueError(f"Invalid mode '{mode}'. Supported modes are: {', '.join(MODES)}")

        super().__init__(fn=self.finalize, tool_name=self.TOOL_NAME, verbose=verbose)
        self.llm = llm
        self.mode = mode
        self.query_refinement_template = query_refinement_template or QUERY_REFINEMENT_TEMPLATE
        self.use_substitution = use_substitution

    def finalize(self, ctx: Context) -> str:
        """
        Finalizes the meal plan recommendation by looking at the context, collecting the meal plan and concatenating it
        with the explanation.
        """
        current_state = RecipeState.from_context(ctx)

        # Fetch data from the context
        meal_plan: MealPlan = current_state.get_meal_plan_recommendation()
        meal_plan_substitution: MealPlanSubstitution = current_state.get_substituted_meal_plan_recommendation()
        explanation: str = current_state.get_meal_plan_explanation()

        # Combine the meal plan and explanation
        if self.mode == MODE_SIMPLE:
            if self.use_substitution and meal_plan_substitution:
                final_recommendation = f"{explanation}\n\nYour meal plan:\n{meal_plan_substitution}"
            else:
                if meal_plan:
                    meal_plan_str = "\n\nYour meal plan:\n" + meal_plan.get_partial_recommendation_str(features=[
                        'title', 'description', 'serves', 'directions', 'ingredient_quantities', 'servingSize [g]'
                    ])
                elif meal_plan_substitution:
                    meal_plan_str = "\n\nYour meal plan:\n" + str(meal_plan_substitution)
                else:
                    meal_plan_str = ""
                final_recommendation = f"{explanation}{meal_plan_str}"
        elif self.mode == MODE_LLM_REFINED:  # Use the LLM to refine the final recommendation
            # Fill in template
            if self.use_substitution and meal_plan_substitution:
                prompt = self.query_refinement_template.format(
                    meal_plan=meal_plan_substitution,
                    explanation=explanation
                )
            else:
                if meal_plan:
                    meal_plan_str = "\n\nYour meal plan:\n" + str(meal_plan)
                elif meal_plan_substitution:
                    meal_plan_str = "\n\nYour meal plan:\n" + str(meal_plan_substitution)
                else:
                    meal_plan_str = "No meal plan recommendation available."
                prompt = self.query_refinement_template.format(
                    meal_plan=meal_plan_str,
                    explanation=explanation
                )

            # Call the LLM
            final_recommendation = self.llm.complete(prompt).text

            # Update the llm usage count
            current_state.increment_llm_counter(parsing_inc=0, sequential_inc=1, total_inc=1)
            current_state.save_to_context()
        else:
            raise ValueError(f"Invalid mode '{self.mode}'. Supported modes are: {', '.join(MODES)}")

        # Update the context with the final recommendation
        current_state.set_final_recommendation(final_recommendation)
        current_state.save_to_context()

        return final_recommendation
