from llama_index.core.base.response.schema import RESPONSE_TYPE
from llama_index.core.schema import QueryType
from llama_index.core.tools import QueryEngineTool, FunctionTool
from llama_index.core.tools.function_tool import AsyncCallable
from typing import Callable, Any

from foodrag.food_rag.query_engines.rag import RecipeRAGQueryEngine

TOOL_NAME = "recipe_tool"


class RecipeToolBase:
    """
    Base class for recipe tools. Extend this class to create a new recipe tool. If the context is called by the
    respective fn function, it must be done as first argument (excluding self).
    """

    def __init__(self, fn: Callable[..., Any] | AsyncCallable, tool_name: str = None, verbose: bool = True):
        """
        Initialize the recipe tool.

        :param fn: The function to run the tool.
        :param tool_name: The name of the tool. If None, uses the default tool name.
        :param verbose: Whether to print debug information.
        """
        self.fn = fn
        self.tool_name = tool_name or TOOL_NAME
        self.verbose = verbose

    def to_tool(self, verbose=False) -> FunctionTool:
        """
        Convert this class into a LlamaIndex FunctionTool.
        :param verbose: Whether to print debug information. Defaults sets verbose of the whole tool to False.
        :return: Configured FunctionTool instance.
        :rtype: FunctionTool
        """
        self.verbose = verbose

        return FunctionTool.from_defaults(fn=self.fn, name=self.tool_name)  # fn can be both a sync and async function


class RecipeQueryToolBase:
    """
    Base class for recipe tools which use a recipe query engine. Classes that inherit from this class should
    extend/override the run method as well as the _get_prompts method.

    NOTE: DOES NOT USE THE CONTEXT OBJECT, AND THEREFORE NO RECIPE STATE INCLUDING THE LLM USAGE COUNTER.
    """

    def __init__(self, recipe_query_engine: RecipeRAGQueryEngine, tool_name: str = None, verbose: bool = True):
        """
        Initialize the agent with a query engine and a language model.

        :param RecipeRAGQueryEngine recipe_query_engine: The recipe query_engine.
        :param tool_name: The name of the query engine tool. If None, uses the default tool name.
        :param verbose: Whether to print debug information.
        """
        self.recipe_query_engine = recipe_query_engine
        self.tool_name = tool_name or TOOL_NAME
        self.verbose = verbose

    def query(self, str_or_query_bundle: QueryType) -> RESPONSE_TYPE:
        """
        Run the query engine tool on the given query.
        Note: This method is NOT used in the QueryEngineTool. The QueryEngineTool directly calls the query method of the
         query engine.

        :param str str_or_query_bundle: The query to run the tool on.
        :return: The response from the query engine tool.
        :rtype: str
        """
        return self.recipe_query_engine.query(str_or_query_bundle)

    def to_tool(self, verbose=False) -> QueryEngineTool:
        """
        Convert this class into a LlamaIndex QueryEngineTool.
        :param verbose: Whether to print debug information. Defaults sets verbose of the whole tool to False.
        :return: Configured QueryEngineTool instance.
        :rtype: QueryEngineTool
        """
        self.verbose = verbose

        return QueryEngineTool.from_defaults(query_engine=self.recipe_query_engine, name=self.tool_name)
