import pandas as pd
from llama_index.core import Settings
from llama_index.core.agent import ReActChatFormatter
from llama_index.core.agent.workflow import FunctionAgent, ReActAgent
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.llms import LLM
from llama_index.core.memory import BaseMemory, ChatMemoryBuffer
from llama_index.core.memory.types import BaseChatStoreMemory
from typing import (
    Optional, List, Type, ClassVar,
)

from foodrag.food_rag.tools import substitution_tools
from foodrag.food_rag.query_engines.rag import RecipeRAGQueryEngine
from foodrag.food_rag.retriever.df_retriever_base import DataFrameRetrieverBase
from foodrag.food_rag.tools import explanation_tools
from foodrag.food_rag.tools import generalization_tools, finalization_tool
from foodrag.food_rag.tools import ingredient_tools, validation_tools, meal_plan_tools, recipe_tools

MERGED_ENHANCEMENT_AGENT_PROMPT = """
You are the EnhancementAgent that enhances, improves, and explains meal plans.
You might improve the meal plan to fit better the user's requirements.
You should explain why the recommended meal plan fits the user's question/requirements.
"""

class HeuristicMealPlanReActAgent(ReActAgent):
    """Agent class that recommends meal plans, can validate them, explain them, but is also able to use its own
     reasoning."""

    SYSTEM_PROMPT: ClassVar[str] = ("""
You are a a meal plan recommender. Follow this heuristic:

1) Recommend the meal plan (meal_plan_recommendation_tool).
2) Validate the meal plan (meal_plan_validation_tool).
3) Generate an explanation for the meal plan (explanation_tool).
4) Provide the validated meal plan recommendation with an explanation.

General rules:
- You can use any tool at any time.
- You can also use your own reasoning. 
- Do but do not invent new recipes.
""")

    NAME: ClassVar[str] = "HeuristicMealPlanReActAgent"

    DESCRIPTION: ClassVar[str] = "Useful for recommending, explaining, and validating meal plans and/or recipes."

    def __init__(self, hummus_data, cleaned_recipes_df, retriever: DataFrameRetrieverBase,
                 query_engine: RecipeRAGQueryEngine,
                 embed_model: BaseEmbedding, ingredient_embeddings=[], food_product_embeddings=[],
                 food_on_embeddings=[], name: str = NAME, description: str = DESCRIPTION,
                 system_prompt: str = SYSTEM_PROMPT, formatter: ReActChatFormatter = None, llm: Optional[LLM] = None,
                 tools: Optional[List] = None, can_handoff_to: Optional[List[str]] = None, memory: BaseMemory = None,
                 memory_cls: Type[BaseMemory] = ChatMemoryBuffer, chat_history: Optional[List[ChatMessage]] = None,
                 **kwargs):
        """
        A ReActAgent that automatically sets the context to a heuristic prompt and allows the agent to use its own
        reasoning.
        """

        llm = llm or Settings.llm
        if memory is None:
            memory = memory_cls.from_defaults(
                chat_history=chat_history or [], llm=llm
            )

        if formatter is None:
            formatter = ReActChatFormatter.from_defaults(
                context=system_prompt, observation_role=MessageRole.USER
            )

        if tools is None:
            # Provide any default tools
            tools = [recipe_tools.RecipeRecommendationTool(recipe_query_engine=query_engine).to_tool(),
                     recipe_tools.RecipeRetrievalTool(recipe_retriever=retriever, llm_calls=1).to_tool(),
                     meal_plan_tools.MealPlanRecommendationTool(llm=llm, recipe_query_engine=query_engine,
                                                                retrieve_only=False,
                                                                time="morning", date="13.04.2025").to_tool(),
                     meal_plan_tools.MealPlanUpdateTool().to_tool(),
                     generalization_tools.RecipeQuestionGeneralizationTool(llm=llm).to_tool(),
                     ingredient_tools.IngredientSubstitutionTool(hummus_data=hummus_data, llm=llm,
                                                                 embed_model=embed_model,
                                                                 ingredient_embeddings=ingredient_embeddings,
                                                                 food_product_embeddings=food_product_embeddings,
                                                                 food_on_embeddings=food_on_embeddings,
                                                                 retrieving_mode=ingredient_tools.DOC_2_VEC).to_tool(),
                     validation_tools.RecipeValidationTool(llm=llm, recipe_df=cleaned_recipes_df).to_tool(),
                     validation_tools.MealPlanValidationTool(llm=llm, recipe_df=cleaned_recipes_df).to_tool(),
                     explanation_tools.ExplanationTool(llm=llm, hummus_data=hummus_data).to_tool(),
                     finalization_tool.FinalizeRecommendationTool(llm=llm,
                                                                  mode=finalization_tool.MODE_SIMPLE).to_tool()]

        if can_handoff_to is None:
            can_handoff_to = []

        super().__init__(
            name=name,
            description=description,
            system_prompt=system_prompt,
            formatter=formatter,
            llm=llm,
            tools=tools,
            can_handoff_to=can_handoff_to,
            memory=memory,
            **kwargs
        )


class RecommendationAgent(FunctionAgent):
    """Agent class that recommends meal plans and/or recipes.

    Uses native function calling (:class:`FunctionAgent`) rather than text-based
    ReAct: Llama-3.3-70B on DeepInfra does not reliably emit the ReAct
    "Thought/Action/Action Input" format, so tool calls and handoffs were being
    dropped. Native ``tool_calls`` (incl. the handoff tool) are far more reliable.
    """

    SYSTEM_PROMPT: ClassVar[str] = (
        "You are the RecommendationAgent that can recommend meal plans. "
        "You MUST call the meal-plan recommendation tool to produce a plan; do not "
        "answer from your own knowledge. "
        "Once your meal plan recommendation is satisfied, hand off control "
        "to the ValidationAgent to evaluate the meal plan. "
        "You should have at least some recommendations before handing off control "
        "to the ValidationAgent."
    )

    NAME: ClassVar[str] = "RecommendationAgent"

    DESCRIPTION: ClassVar[str] = "Useful for recommending meal plans and/or recipes."

    def __init__(
            self,
            query_engine: RecipeRAGQueryEngine,
            name: str = NAME,
            description: str = DESCRIPTION,
            system_prompt: str = SYSTEM_PROMPT,
            llm: Optional[LLM] = None,
            tools: Optional[List] = None,
            can_handoff_to: Optional[List[str]] = None,
            time: Optional[str] = "now",
            date: Optional[str] = "today",
            full_day: bool = False,
            single_meal: Optional[str] = None,
            **kwargs
    ):
        """``time``/``date`` are forwarded to the meal-plan recommendation tool so the
        live H1 workflow can pin (or default to "now"/"today") the recommendation time.
        Whatever is resolved is captured into RecipeState and persisted per record.

        ``full_day`` (default False) is forwarded to the meal-plan recommendation tool so
        the H1 chain requests ALL FOUR meal types (breakfast/lunch/dinner/snack) — a full
        4-slot plan — instead of only the meals relevant to the current time.

        ``single_meal`` (optional meal type) restricts the plan to exactly ONE slot,
        skipping the LLM meal-split — much faster (fewer meals to validate/explain)."""
        llm = llm or Settings.llm

        if tools is None:
            # Provide any default tools needed for meal-plan or recipe recommendations
            tools = [
                meal_plan_tools.MealPlanRecommendationTool(
                    llm=llm, recipe_query_engine=query_engine, retrieve_only=True, time=time, date=date,
                    full_day=full_day, single_meal=single_meal
                ).to_tool(),
            ]

        if can_handoff_to is None:
            can_handoff_to = ["ValidationAgent"]

        super().__init__(
            name=name,
            description=description,
            system_prompt=system_prompt,
            llm=llm,
            tools=tools,
            can_handoff_to=can_handoff_to,
            **kwargs
        )


class ValidationAgent(FunctionAgent):
    """Agent class that validates meal plans (native function calling)."""

    SYSTEM_PROMPT: ClassVar[str] = (
        "You are the ValidationAgent that validates meal plans. "
        "You MUST call the recipe correction tool to validate and, if needed, correct "
        "the current meal plan before handing off. "
        "After the tool runs, hand off control: if the plan is still INVALID hand off "
        "to the SubstitutionAgent; otherwise hand off to the ExplanationAgent."
    )

    NAME: ClassVar[str] = "ValidationAgent"

    DESCRIPTION: ClassVar[str] = "Useful for checking if a meal plan is a valid recommendation."

    def __init__(
            self,
            query_engine: RecipeRAGQueryEngine,
            recipe_df: pd.DataFrame,
            name: str = NAME,
            description: str = DESCRIPTION,
            system_prompt: str = SYSTEM_PROMPT,
            llm: Optional[LLM] = None,
            tools: Optional[List] = None,
            can_handoff_to: Optional[List[str]] = None,
            max_correction_rounds: Optional[int] = None,
            **kwargs
    ):
        llm = llm or Settings.llm
        if tools is None:
            # Provide any default validation tools. ``max_correction_rounds`` caps the
            # generalization loop so hard constraints don't run to a timeout.
            tools = [
                validation_tools.CorrectionTool(
                    llm=llm, recipe_df=recipe_df, recipe_query_engine=query_engine,
                    max_correction_rounds=max_correction_rounds).to_tool(),
            ]
        if can_handoff_to is None:
            can_handoff_to = ["ExplanationAgent", "SubstitutionAgent"]

        super().__init__(
            name=name,
            description=description,
            system_prompt=system_prompt,
            llm=llm,
            tools=tools,
            can_handoff_to=can_handoff_to,
            **kwargs
        )


class SubstitutionAgent(FunctionAgent):
    """Agent class that enhances or improves meal plans (native function calling)."""

    SYSTEM_PROMPT: ClassVar[str] = (
        "You are the SubstitutionAgent that enhances or improves meal plans. "
        "Call the substitution tool to improve the current meal plan, then hand off "
        "control to the ExplanationAgent to explain the meal plan."
    )

    NAME: ClassVar[str] = "SubstitutionAgent"

    DESCRIPTION: ClassVar[str] = "Useful for enhancing meal plans."

    def __init__(
            self,
            name: str = NAME,
            description: str = DESCRIPTION,
            system_prompt: str = SYSTEM_PROMPT,
            llm: Optional[LLM] = None,
            tools: Optional[List] = None,
            can_handoff_to: Optional[List[str]] = None,
            **kwargs
    ):
        llm = llm or Settings.llm
        if tools is None:
            # Provide any default substitution/improvement tools
            tools = [
                substitution_tools.SubstitutionTool(llm=llm).to_tool()
            ]
        if can_handoff_to is None:
            can_handoff_to = ["ExplanationAgent"]

        super().__init__(
            name=name,
            description=description,
            system_prompt=system_prompt,
            llm=llm,
            tools=tools,
            can_handoff_to=can_handoff_to,
            **kwargs
        )


class ExplanationAgent(FunctionAgent):
    """Agent class that explains recommended meal plans (native function calling)."""

    SYSTEM_PROMPT: ClassVar[str] = (
        "You are the ExplanationAgent that explains recommended meal plans. "
        "You MUST call the explanation tool to generate the explanation of why the "
        "recommended meal plan fits the user's question/requirements, then hand off "
        "control to the FinalizationAgent."
    )

    NAME: ClassVar[str] = "ExplanationAgent"

    DESCRIPTION: ClassVar[str] = "Useful for explaining meal plans."

    def __init__(
            self,
            hummus_data,
            name: str = NAME,
            description: str = DESCRIPTION,
            system_prompt: str = SYSTEM_PROMPT,
            llm: Optional[LLM] = None,
            tools: Optional[List] = None,
            can_handoff_to: Optional[List[str]] = None,
            **kwargs
    ):
        llm = llm or Settings.llm
        if tools is None:
            # Provide any default explanation tools
            tools = [
                explanation_tools.ExplanationTool(llm=llm,hummus_data=hummus_data).to_tool(),
            ]
        # ExplanationAgent is the last in the chain, so by default it may not hand off to others.
        if can_handoff_to is None:
            can_handoff_to = ["FinalizationAgent"]

        super().__init__(
            name=name,
            description=description,
            system_prompt=system_prompt,
            llm=llm,
            tools=tools,
            can_handoff_to=can_handoff_to,
            **kwargs
        )

class FinalizationAgent(FunctionAgent):
    """Agent class that finalizes recommended meal plans (native function calling)."""

    SYSTEM_PROMPT: ClassVar[str] = (
        "You are the FinalizationAgent. You MUST call the finalize meal-plan "
        "recommendation tool exactly once to assemble the final recommendation "
        "(meal plan + explanation), then stop."
    )

    NAME: ClassVar[str] = "FinalizationAgent"

    DESCRIPTION: ClassVar[str] = "Useful for finalizing meal plan recommendations."

    def __init__(
            self,
            name: str = NAME,
            description: str = DESCRIPTION,
            system_prompt: str = SYSTEM_PROMPT,
            llm: Optional[LLM] = None,
            tools: Optional[List] = None,
            can_handoff_to: Optional[List[str]] = None,
            **kwargs
    ):
        llm = llm or Settings.llm
        if tools is None:
            # Provide any default explanation tools
            tools = [
                finalization_tool.FinalizeRecommendationTool(llm=llm, mode=finalization_tool.MODE_SIMPLE).to_tool()
            ]
        # FinalizationAgent is the last in the chain, so by default it may not hand off to others.
        if can_handoff_to is None:
            can_handoff_to = []

        super().__init__(
            name=name,
            description=description,
            system_prompt=system_prompt,
            llm=llm,
            tools=tools,
            can_handoff_to=can_handoff_to,
            **kwargs
        )