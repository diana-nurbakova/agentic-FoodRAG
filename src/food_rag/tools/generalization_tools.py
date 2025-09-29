from llama_index.core import PromptTemplate
from llama_index.core.llms import LLM
from llama_index.core.workflow import Context
from pydantic import Field
from pydantic.fields import FieldInfo

from src.food_rag.tools.tool_base import RecipeToolBase
from src.food_rag.utils.recipe_context import RecipeState


class RecipeQuestionGeneralizationTool(RecipeToolBase):
    """
    A tool that generalizes recipe queries.
    """
    TOOL_NAME = "recipe_question_generalization_tool"

    STEPS = {
        1: "- Keep all ingredients if available, recipe name, and exact nutritional requirements if available.\n",
        2: "- Keep all ingredients if available, recipe name, and generalized nutritional requirements if available (high, medium, low).\n",
        3: "- Keep all ingredients if available, and the recipe name.\n"
           "- Discard all nutritional requirements.\n",
        4: "- Keep the recipe name, and generalized ingredients if available (e.g., 'chicken breast' -> 'chicken').\n",
        5: "- Keep the recipe name.\n"
           "- Discard all other requirements.\n"
           "- Remove redundant requirements.\n"
    }

    TEMPLATE = PromptTemplate(
        "Reformulate the question below using the following rules:\n"
        "{instructions}"
        "- Keep any excluded or undesired ingredients (e.g., 'allergens such as wheat', 'dairy', 'fish'). Do not remove those or alter them.\n"
        "- Do not add new information.\n"
        "- Do not add new ingredients.\n"
        "- DO NOT ANSWER THE QUESTION.\n"
        "- ONLY RETURN THE REFORMULATED QUESTION.\n"
        "Question (to be reformulated): '{question}'\n"
        "Reformulated question:"
    )

    MAX_GENERALIZATION_LEVEL = len(STEPS)

    def __init__(self, llm: LLM, template: PromptTemplate = None, verbose: bool = True):
        """
        Initialize the tool.

        :param LLM llm: The language model instance.
        :param PromptTemplate template: The template for the generalization steps. If None, the default template is used.
        :param bool verbose: Whether to print the generalization step.
        :raises ValueError: If the step is not between 1 and 5.
        """
        self.llm = llm
        self.template = template or self.TEMPLATE

        super().__init__(fn=self.generalize_question, tool_name=self.TOOL_NAME, verbose=verbose)

    def generalize_question(self, ctx: Context, question: str = Field(
        description="The original question which shall be generalized."),
                            generalization_level: int = Field(
                                "The level of the generalization between 1 and 5. Start with level 1.")) -> str:
        """
        Simplifies a recipe question if it was too specific and not usable for retrieving any recipe before. Tool can be
        used iteratively to generalize the question step by step. The level of generalization ranges from 1 to 5. Start
        with level 1. This tool does not answer the question but provides a generalized version of it.
        """
        if isinstance(question, FieldInfo) or isinstance(generalization_level, FieldInfo):
            raise ValueError("The 'question' parameter must be provided as string, the 'generalization_level'"
                             " parameter must be provided as an integer.")

        if generalization_level < 1 or generalization_level > 5:
            raise ValueError(f"Cannot generalize question further. Invalid level {generalization_level}. "
                             f"The step must be between 1 and 5.")

        current_state = RecipeState.from_context(ctx)

        # Get the generalization instruction for the given question
        instructions = self.STEPS.get(generalization_level)
        generalization_instruction = self.TEMPLATE.format(instructions=instructions, question=question)

        if self.verbose:
            print(f"Generalization instruction at step {generalization_level}:\n''{generalization_instruction}''"
                  f"\n--------------\n")

        # Use the llm to generate a question from the generalization instruction
        new_question = self.llm.complete(generalization_instruction)

        # Update the llm usage count
        current_state.increment_llm_counter(parsing_inc=0, sequential_inc=1, total_inc=1)
        current_state.save_to_context()

        if self.verbose:
            print(f"Generalized question at step {generalization_level}:\n''{new_question.text}''\n--------------\n")

        return new_question.text
