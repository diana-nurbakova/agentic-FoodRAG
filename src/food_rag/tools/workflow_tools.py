import pandas as pd
from llama_index.core import Response
from llama_index.core.llms import LLM
from llama_index.core.workflow import Context
from typing import List

from src.food_rag.utils.recipe_context import RecipeState
from src.food_rag.query_engines.rag import RecipeRAGQueryEngine
from src.food_rag.tools import meal_plan_tools, validation_tools, substitution_tools, explanation_tools, \
    finalization_tool
from src.food_rag.tools.tool_base import RecipeToolBase


class MealPlanWorkflowTool(RecipeToolBase):
    """
    A tool that integrates multiple recipe tools to create the complete meal plan workflow.
    """

    TOOL_NAME = "meal_plan_workflow_tool"

    def __init__(self, hummus_data, llm: LLM, recipe_df: pd.DataFrame, recipe_query_engine: RecipeRAGQueryEngine,
                 verbose: bool = True):
        """
        Initialize the tool.

        :param hummus_data: The hummus data.
        :param llm: The language model instance.
        :param recipe_df: The (cleaned) Pandas DataFrame containing known recipes.
        :param recipe_query_engine: The query engine for the recipes.
        :param verbose: If True, prints the correction result.
        """
        super().__init__(fn=self.run_meal_plan_workflow, tool_name=self.TOOL_NAME, verbose=verbose)
        self.llm = llm
        self.recipe_df = recipe_df

        # Init all the tools
        self.meal_plan_tool = meal_plan_tools.MealPlanRecommendationTool(llm=llm,
                                                                         recipe_query_engine=recipe_query_engine,
                                                                         retrieve_only=True, time="morning",
                                                                         date="13.04.2025")
        self.validation_tool = validation_tools.CorrectionTool(llm=llm, recipe_df=recipe_df,
                                                               recipe_query_engine=recipe_query_engine)
        self.substitution_tool = substitution_tools.SubstitutionTool(llm=llm)
        self.explanation_tool = explanation_tools.ExplanationTool(llm=llm, hummus_data=hummus_data)
        self.finalization_tool = finalization_tool.FinalizeRecommendationTool(llm=llm,
                                                                              mode=finalization_tool.MODE_SIMPLE)

    def run_meal_plan_workflow(self, ctx: Context, question: str, objectives: List[str] = ["healthier"]) -> str:
        """
        Run the meal plan workflow.
        """
        self.meal_plan_tool.recommend_meal_plan(ctx, question)
        self.validation_tool.correct_recipes(ctx)
        self.substitution_tool.substitute(ctx, objectives if objectives else ["healthier"])
        self.explanation_tool.explain(ctx)
        return self.finalization_tool.finalize(ctx)


class MealPlanShortWorkflowTool(RecipeToolBase):
    """
    A tool that integrates one simple recipe tool to create the complete meal plan workflow.
    """

    TOOL_NAME = "meal_plan_short_workflow_tool"

    def __init__(self, llm: LLM, recipe_df: pd.DataFrame, recipe_query_engine: RecipeRAGQueryEngine,
                 verbose: bool = True):
        """
        Initialize the tool.

        :param llm: The language model instance.
        :param recipe_df: The (cleaned) Pandas DataFrame containing known recipes.
        :param recipe_query_engine: The query engine for the recipes.
        :param verbose: If True, prints the correction result.
        """
        super().__init__(fn=self.run_meal_plan_workflow, tool_name=self.TOOL_NAME, verbose=verbose)
        self.llm = llm
        self.recipe_df = recipe_df

        # Init all the tools
        self.meal_plan_tool = meal_plan_tools.MealPlanRecommendationTool(llm=llm,
                                                                         recipe_query_engine=recipe_query_engine,
                                                                         retrieve_only=True, time="morning",
                                                                         date="13.04.2025")

    def run_meal_plan_workflow(self, ctx: Context, question: str) -> str:
        """
        Run the meal plan workflow.
        """
        meal_plan = self.meal_plan_tool.recommend_meal_plan(ctx, question)
        return meal_plan.get_recommendation_str()

class MealPlanRAGTool(RecipeToolBase):
    """
    A tool that integrates the RAG query engine to create a meal plan.
    """

    TOOL_NAME = "meal_plan_rag_tool"

    def __init__(self, llm: LLM, recipe_query_engine: RecipeRAGQueryEngine, verbose: bool = True):
        """
        Initialize the tool.

        :param llm: The language model instance.
        :param recipe_query_engine: The query engine for the recipes.
        :param verbose: If True, prints the correction result.
        """
        super().__init__(fn=self.run_meal_plan_rag, tool_name=self.TOOL_NAME, verbose=verbose)
        self.llm = llm
        self.recipe_query_engine: RecipeRAGQueryEngine = recipe_query_engine

    def run_meal_plan_rag(self, ctx: Context, question: str) -> str:
        """
        Run the meal plan RAG workflow.
        """
        current_state = RecipeState.from_context(ctx)

        response: Response = self.recipe_query_engine.query(question)

        current_state.increment_llm_counter(sequential_inc=self.recipe_query_engine.llm_calls,
                                            total_inc=self.recipe_query_engine.llm_calls)
        current_state.save_to_context()

        return response.response