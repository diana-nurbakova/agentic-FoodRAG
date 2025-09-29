import pandas as pd
from llama_index.core import Settings
from llama_index.core.agent import ReActChatFormatter
from llama_index.core.agent.workflow import ReActAgent
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.llms import LLM
from llama_index.core.memory import BaseMemory, ChatMemoryBuffer
from llama_index.core.memory.types import BaseChatStoreMemory
from typing import (
    Optional, List, Type, ClassVar,
)

from src.food_rag.tools import substitution_tools
from src.food_rag.query_engines.rag import RecipeRAGQueryEngine
from src.food_rag.retriever.df_retriever_base import DataFrameRetrieverBase
from src.food_rag.tools import explanation_tools
from src.food_rag.tools import generalization_tools, finalization_tool
from src.food_rag.tools import ingredient_tools, validation_tools, meal_plan_tools, recipe_tools

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
                context=system_prompt, observation_role=MessageRole.TOOL
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


class RecommendationAgent(ReActAgent):
    """Agent class that recommends meal plans and/or recipes."""

    SYSTEM_PROMPT: ClassVar[str] = (
        "You are the RecommendationAgent that can recommend meal plans. "
        "Once your meal plan recommendation is satisfied, you should hand off control "
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
            formatter: ReActChatFormatter = None,
            llm: Optional[LLM] = None,
            tools: Optional[List] = None,
            can_handoff_to: Optional[List[str]] = None,
            memory: BaseMemory = None,
            memory_cls: Type[BaseMemory] = ChatMemoryBuffer,
            chat_history: Optional[List[ChatMessage]] = None,
            **kwargs
    ):
        llm = llm or Settings.llm
        if memory is None:
            memory = memory_cls.from_defaults(
                chat_history=chat_history or [], llm=llm
            )

        if formatter is None:
            formatter = ReActChatFormatter.from_defaults(
                context=system_prompt, observation_role=MessageRole.TOOL
            )

        if tools is None:
            # Provide any default tools needed for meal-plan or recipe recommendations
            tools = [
                meal_plan_tools.MealPlanRecommendationTool(
                    llm=llm, recipe_query_engine=query_engine, retrieve_only=True, time="morning", date="13.04.2025"
                ).to_tool(),
            ]

        if can_handoff_to is None:
            can_handoff_to = ["ValidationAgent"]

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


class ValidationAgent(ReActAgent):
    """Agent class that validates meal plans."""

    SYSTEM_PROMPT: ClassVar[str] = (
        "You are the ValidationAgent that validates meal plans. "
        "Once the validation is done, if it is INVALID go back to the RecommendationAgent to restart. "
        "If the validation is VALID go to the ExplanationAgent, else you should hand off control to the "
        "SubstitutionAgent."
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
            formatter: ReActChatFormatter = None,
            llm: Optional[LLM] = None,
            tools: Optional[List] = None,
            can_handoff_to: Optional[List[str]] = None,
            memory: BaseMemory = None,
            memory_cls: Type[BaseMemory] = ChatMemoryBuffer,
            chat_history: Optional[List[ChatMessage]] = None,
            **kwargs
    ):
        llm = llm or Settings.llm
        if memory is None:
            memory = memory_cls.from_defaults(
                chat_history=chat_history or [], llm=llm
            )
        if formatter is None:
            formatter = ReActChatFormatter.from_defaults(
                context=system_prompt, observation_role=MessageRole.TOOL
            )
        if tools is None:
            # Provide any default validation tools
            tools = [
                validation_tools.CorrectionTool(llm=llm, recipe_df=recipe_df,
                                                recipe_query_engine=query_engine).to_tool(),
            ]
        if can_handoff_to is None:
            can_handoff_to = ["ExplanationAgent", "SubstitutionAgent"]

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


class SubstitutionAgent(ReActAgent):
    """Agent class that enhances or improves meal plans."""

    SYSTEM_PROMPT: ClassVar[str] = (
        "You are the SubstitutionAgent that enhances or improves meal plans. "
        "Once the meal plan is satisfied, you should hand off control to the ExplanationAgent "
        "to explain the meal plan."
    )

    NAME: ClassVar[str] = "SubstitutionAgent"

    DESCRIPTION: ClassVar[str] = "Useful for enhancing meal plans."

    def __init__(
            self,
            name: str = NAME,
            description: str = DESCRIPTION,
            system_prompt: str = SYSTEM_PROMPT,
            formatter: ReActChatFormatter = None,
            llm: Optional[LLM] = None,
            tools: Optional[List] = None,
            can_handoff_to: Optional[List[str]] = None,
            memory: BaseMemory = None,
            memory_cls: Type[BaseMemory] = ChatMemoryBuffer,
            chat_history: Optional[List[ChatMessage]] = None,
            **kwargs
    ):
        llm = llm or Settings.llm
        if memory is None:
            memory = memory_cls.from_defaults(
                chat_history=chat_history or [], llm=llm
            )
        if formatter is None:
            formatter = ReActChatFormatter.from_defaults(
                context=system_prompt, observation_role=MessageRole.TOOL
            )
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
            formatter=formatter,
            llm=llm,
            tools=tools,
            can_handoff_to=can_handoff_to,
            memory=memory,
            **kwargs
        )


class ExplanationAgent(ReActAgent):
    """Agent class that explains recommended meal plans."""

    SYSTEM_PROMPT: ClassVar[str] = (
        "You are the ExplanationAgent that explains recommended meal plans. "
        "You should explain why the recommended meal plan fits the user's question/requirements."
    )

    NAME: ClassVar[str] = "ExplanationAgent"

    DESCRIPTION: ClassVar[str] = "Useful for explaining meal plans."

    def __init__(
            self,
            hummus_data,
            name: str = NAME,
            description: str = DESCRIPTION,
            system_prompt: str = SYSTEM_PROMPT,
            formatter: ReActChatFormatter = None,
            llm: Optional[LLM] = None,
            tools: Optional[List] = None,
            can_handoff_to: Optional[List[str]] = None,
            memory: BaseChatStoreMemory = None,
            memory_cls: Type[BaseMemory] = ChatMemoryBuffer,
            chat_history: Optional[List[ChatMessage]] = None,
            **kwargs
    ):
        llm = llm or Settings.llm
        if memory is None:
            memory = memory_cls.from_defaults(
                chat_history=chat_history or [], llm=llm
            )
        if formatter is None:
            formatter = ReActChatFormatter.from_defaults(
                context=system_prompt, observation_role=MessageRole.TOOL
            )
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
            formatter=formatter,
            llm=llm,
            tools=tools,
            can_handoff_to=can_handoff_to,
            memory=memory,
            **kwargs
        )

class FinalizationAgent(ReActAgent):
    """Agent class that finalizes recommended meal plans."""

    SYSTEM_PROMPT: ClassVar[str] = (
        "You are the FinalizationAgent that explains recommended meal plans. "
    )

    NAME: ClassVar[str] = "FinalizationAgent"

    DESCRIPTION: ClassVar[str] = "Useful for finalizing meal plan recommendations."

    def __init__(
            self,
            name: str = NAME,
            description: str = DESCRIPTION,
            system_prompt: str = SYSTEM_PROMPT,
            formatter: ReActChatFormatter = None,
            llm: Optional[LLM] = None,
            tools: Optional[List] = None,
            can_handoff_to: Optional[List[str]] = None,
            memory: BaseChatStoreMemory = None,
            memory_cls: Type[BaseMemory] = ChatMemoryBuffer,
            chat_history: Optional[List[ChatMessage]] = None,
            **kwargs
    ):
        llm = llm or Settings.llm
        if memory is None:
            memory = memory_cls.from_defaults(
                chat_history=chat_history or [], llm=llm
            )
        if formatter is None:
            formatter = ReActChatFormatter.from_defaults(
                context=system_prompt, observation_role=MessageRole.TOOL
            )
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
            formatter=formatter,
            llm=llm,
            tools=tools,
            can_handoff_to=can_handoff_to,
            memory=memory,
            **kwargs
        )