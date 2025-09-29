from llama_index.core import PromptTemplate
from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.llms import LLM
from llama_index.core.memory.types import BaseChatStoreMemory
from llama_index.core.workflow import Context
from typing import List

from src.food_rag.meal_plan.meal_plan_structures import MealPlanSubstitution
from src.food_rag.meal_plan.meal_plan_structures import MealPlan
from src.food_rag.tools.tool_base import RecipeToolBase
from src.food_rag.utils.recipe_context import RecipeState


class ExplanationTool(RecipeToolBase):
    """
    A tool that generates a concise explanation for a final meal plan recommendation.
    """

    TOOL_NAME = "explanation_tool"

    EXPLANATION_TEMPLATE = PromptTemplate(
        f"Write a clear and persuasive explanation of the recommended meal plan:\n"
        "- Focus on why this plan matches the user's original question.\n"
        "- Use relevant context from the history if it supports your reasoning, but do not mention the history itself.\n"
        "- Only refer to information that justifies the **final** recommendation.\n"
        "- Do not explain how the recommendation was created or evolved.\n"
        "- Do not suggest any new or modified meals—explain only what is already recommended.\n"
        "- Make the explanation suitable for a user who is deciding whether to follow the plan.\n"
        "- Avoid irrelevant or technical details. Be concise and user-friendly.\n"
        "- Do not include specific names, timestamps, or sequences from the history.\n"
        "- Example:\n"
        "  User question:\n"
        "  'I want to eat fish with a minimal fsa_score of 0.06, totalFat content between 30–300g, "
        "maximum cholesterol level of 356.06mg, and excluding Californian tags.'\n\n"
        "  Explanation:\n"
        "  'The recommended meal plan includes a selection of fish-based dishes that meet your specific dietary needs. "
        "Each meal maintains an FSA score of at least 0.06, supporting better nutritional quality. The fat content "
        "is within your requested range of 30 to 300 grams, and the cholesterol levels stay below your 356.06mg limit. "
        "Additionally, all meals exclude any ingredients with Californian tags, ensuring they align with your preferences. "
        "This plan is designed to be both enjoyable and nutritionally balanced.'\n\n\n"
        "Original user question:\n{original_question}\n\n"
        "Recommended meal plan:\n{meal_plan}\n\n"
        "History:\n{history}\n\n"
        "Explanation:"
    )

    def __init__(self, llm: LLM, hummus_data, chat_memory: BaseChatStoreMemory = None,
                 explanation_template: PromptTemplate = None,
                 verbose: bool = True):
        """
        Initialize the explanation tool. Uses the RecipeState History to get the chat history and additionally
        optionally the chat memory.

        :param LLM llm: The language model instance.
        :param hummus_data: The hummus data.
        :param chat_memory: The chat memory. If None, the chat memory is not used.
        :param explanation_template: The template for the explanation. Defaults to the EXPLANATION_TEMPLATE.
        :param verbose: If True, prints the generated explanation to stdout.
        """
        super().__init__(fn=self.explain, tool_name=self.TOOL_NAME, verbose=verbose)
        self.llm = llm
        self.hummus_data = hummus_data
        self.chat_memory = chat_memory
        self.explanation_template = explanation_template or self.EXPLANATION_TEMPLATE

    def set_chat_memory(self, chat_memory: BaseChatStoreMemory) -> None:
        """
        Set the chat memory by the agent.

        :param chat_memory: The chat memory.
        """
        self.chat_memory = chat_memory

    def explain(self, ctx: Context) -> str:
        """ Generates a concise explanation for the final recommended meal plan. """
        current_state = RecipeState.from_context(ctx)
        history: list = current_state.get_history()
        meal_plan: MealPlan = current_state.get_meal_plan_recommendation()
        substituted_meal_plan: MealPlanSubstitution = current_state.get_substituted_meal_plan_recommendation()
        original_question: str = current_state.get_original_user_question()

        # Get and format the chat history from the agent
        if self.chat_memory is not None:
            raw_history: List[ChatMessage] = self.chat_memory.get_all()

            if len(raw_history) != 0:
                chat_history = "\n".join([
                    f"Message {i + 1} (from {msg.role.value}):\n"
                    + "\n".join("   " + line for line in msg.content.splitlines())
                    for i, msg in enumerate(raw_history)
                ])
            else:
                if self.verbose:
                    print("The chat history is empty.")
                chat_history = None
        else:
            if self.verbose:
                print("No chat memory provided. Thus the chat history is empty. You can (optionally) add the chat "
                      "memory post-creation using the 'set_chat_memory()' method.")
            chat_history = None

        # Format the RecipeState history
        if len(history) != 0:
            state_history = "\n".join([f"{i + 1}.\n{msg}\n" for i, msg in enumerate(history)])
        else:
            if self.verbose:
                print("The state history is empty.")
            state_history = ""

        # Combine the chat history and state history
        if chat_history is not None:
            full_history = f"Relevant history:\n{state_history}\n\nFull chat history:\n{chat_history}"
        else:
            full_history = state_history

        # Choose the current recommendation (if substituted meal plan is available take it,
        # otherwise take the original meal plan)
        if not substituted_meal_plan.is_empty():
            recommendation = substituted_meal_plan
        else:
            recommendation = meal_plan

        # Use LLM to generate the explanation
        explanation_prompt = self.explanation_template.format(
            original_question=original_question,
            meal_plan=recommendation,
            history=full_history,
        )

        if self.verbose:
            print(f"Explanation prompt:\n''{explanation_prompt}''\n--------------\n")

        # Call the LLM
        explanation_response = self.llm.complete(explanation_prompt)

        # Update the llm usage count
        current_state.increment_llm_counter(parsing_inc=0, sequential_inc=1, total_inc=1)

        if self.verbose:
            print(f"Explanation:\n''{explanation_response.text}''\n--------------\n")

        current_state.set_meal_plan_explanation(explanation_response.text)
        current_state.update_history_item(explanation_response.text)
        current_state.save_to_context()

        return explanation_response.text
