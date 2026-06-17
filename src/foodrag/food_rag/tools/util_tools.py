from llama_index.core import PromptTemplate
from llama_index.core.llms import LLM
from pydantic import Field
from pydantic.fields import FieldInfo
from llama_index.core.workflow import Context

from foodrag.food_rag.utils.recipe_context import RecipeState
from foodrag.food_rag.tools.tool_base import RecipeToolBase


class JsonExtractorTool(RecipeToolBase):
    """
    A tool that extracts JSON from a string.
    """

    TOOL_NAME = "json_extractor_tool"

    TEMPLATE = PromptTemplate(
        """- Given a text, extract the desired information and return it as json.
        - Only return the json.
        - Format the json like this: ```json{<key>: <value>, ...}```
        
        Desired information:
        {desired_information}
        
        
        Text:
        {text}
        
        
        Return:"""
    )

    DESIRED_INFORMATION = """
        key=title
        The title of the recommended recipe.
        Return a json like this: ```json{"title": <value>}```
    """

    def __init__(self, llm: LLM, desired_information: str = DESIRED_INFORMATION, prompt_template: PromptTemplate = None,
                 verbose=True):
        super().__init__(fn=self.extract_json, tool_name=self.TOOL_NAME, verbose=verbose)

        self.llm = llm
        self.prompt_template = prompt_template or self.TEMPLATE
        self.desired_information = desired_information

    def extract_json(self, ctx: Context,
                     text: str = Field(description="The text from where information shall be extracted.")) -> str:
        """
        Extracts JSON from a string text.
        """
        if isinstance(text, FieldInfo):
            raise ValueError("The 'text' parameter must be provided as string.")

        current_state = RecipeState.from_context(ctx)

        # Generate the prompt
        prompt = self.prompt_template.format(desired_information=self.desired_information, text=text)

        # Use the llm to generate a question from the prompt
        json_str = self.llm.complete(prompt).text

        # Update the llm usage count
        current_state.increment_llm_counter(parsing_inc=1, sequential_inc=1, total_inc=1)
        current_state.save_to_context()

        # Try to extract the json from the response (output between  ```json and ```).
        try:
            return json_str.split('```json')[1].split('```')[0].strip()
        except Exception:
            return ""
