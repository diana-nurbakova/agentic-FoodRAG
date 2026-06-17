"""Pandas output parser but returning a df instead of a str. Original code taken from llama_index-pandas."""

import ast
import logging
import numpy as np
import pandas as pd
import sys
from llama_index.core import QueryBundle, PromptTemplate
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.output_parsers.base import ChainableOutputParser
from llama_index.core.output_parsers.utils import parse_code_markdown
from llama_index.core.schema import NodeWithScore, QueryType
from llama_index.experimental import PandasQueryEngine
from llama_index.experimental.exec_utils import safe_eval, safe_exec
from typing import Any, Dict, Optional, List

from foodrag.food_rag.retriever.df_retriever_base import DataFrameRetrieverBase
from foodrag.food_rag.meal_plan.meal_type import MealType

logger = logging.getLogger(__name__)

DF_COLUMN_DESCRIPTION_DICT = {
    "duration [min]": "The duration in minutes",
    "directions": "The directions or steps to take in order to prepare the recipe",
    "ingredient_quantities": "A list of ingredients and their quantities",
    "serves": "The number of servings or persons the recipe serves",
    "tags": "The corresponding tags of the recipe",
    "average_rating": "The average rating",
    "number_of_ratings": "The number of ratings the recipe has",
    "servingSize [g]": "The size for one portion in grams",
    "calories [kcal]": "The calories in kilocalories",
    "caloriesFromFat [kcal]": "The calories from fat in kilocalories",
    "totalFat [g]": "The total fat in grams",
    "saturatedFat [g]": "The saturated fat in grams",
    "cholesterol [mg]": "The cholesterol in milligrams",
    "sodium [mg]": "The sodium in milligrams",
    "totalCarbohydrate [g]": "The total carbohydrate in grams",
    "dietaryFiber [g]": "The dietary fiber in grams",
    "sugars [g]": "The sugars in grams",
    "protein [g]": "The protein in grams",
    "who_score": "The WHO score, a score between 0 and 1 indicating the healthiness of the recipe",
    "fsa_score": "The FSA score, a score between 0 and 1 indicating the healthiness of the recipe",
    "nutri_score": "The nutritional score, a score between 0 and 1 indicating the healthiness of the recipe",
    "number_of_directions": "The number of directions one has to follow",
    "number_of_ingredients": "The number of ingredients needed",
    "meal_types": "The meal types of the recipe, which can be one or multiple of " + ", ".join(MealType.get_all_str()),
}

INSTRUCTION_STR = """\
1. Convert the query to executable Python code using Pandas.
2. The final line of code should be a Python expression that can be called with the `eval()` function.
3. The code must return a df with all columns of the original df.
4. PRINT ONLY THE EXPRESSION.
5. Do not quote the expression.
"""

PANDAS_PROMPT_TEMPLATE = PromptTemplate(
    """\
You are working with a pandas dataframe in Python.
The name of the dataframe is `df`.
The available columns are the following:
{df_str}

Follow these instructions:
{instruction_str}
Query: {query_str}

Expression: """
)


def default_output_processor(output: str, df: pd.DataFrame, **output_kwargs: Any) -> pd.DataFrame:
    """
    Processes outputs in a default manner.

    :param str output: The output string to process.
    :param pd.DataFrame df: The DataFrame to use in processing.
    :param kwargs: Additional keyword arguments for processing.
    :returns: The processed DataFrame.
    :rtype: pd.DataFrame
    """
    if sys.version_info < (3, 9):
        Exception("Python version must be >= 3.9 to use the pandas output parser.")

    local_vars = {"df": df, "pd": pd}
    global_vars = {"np": np}

    output = parse_code_markdown(output, only_last=True)[0]

    # NOTE: inspired from langchain's tool
    # see langchain.tools.python.tool (PythonAstREPLTool)
    try:
        tree = ast.parse(output)
        module = ast.Module(tree.body[:-1], type_ignores=[])
        safe_exec(ast.unparse(module), {}, local_vars)  # type: ignore
        module_end = ast.Module(tree.body[-1:], type_ignores=[])
        module_end_str = ast.unparse(module_end)  # type: ignore

        if module_end_str.strip("'\"") != module_end_str:
            # if there's leading/trailing quotes, then we need to eval
            # string to get the actual expression
            module_end_str = safe_eval(module_end_str, global_vars, local_vars)

        try:
            output_df = safe_eval(module_end_str, global_vars, local_vars)
            return output_df
        except Exception:
            output_df = custom_eval_pandas(df, module_end_str, global_vars, local_vars,
                                           verbose=output_kwargs.get("verbose", False))
            return output_df
    except Exception:
        # Return an empty DataFrame if an error occurs (but wih the same columns as the original DataFrame)
        return pd.DataFrame(columns=df.columns)


def custom_eval_pandas(df, pandas_instruction_str, global_vars, local_vars, verbose=False):  # UNSAFE!
    # If there's a valid instruction string, use it to query the DataFrame
    if pandas_instruction_str:
        try:
            return eval(pandas_instruction_str, global_vars, local_vars)
        except Exception as e1:
            try:
                return pd.eval(pandas_instruction_str, local_dict=local_vars, global_dict=global_vars)
            except Exception as e2:
                if verbose:
                    print("There was an error running the output as Python code. Could also not eval with pd.eval."
                          f"\nPandas query: {pandas_instruction_str}"
                          f"\nError message (eval): {e1}"
                          f"\nError message (pd.eval): {e2}")

                # Return an empty DataFrame if an error occurs (but wih the same columns as the original DataFrame)
                return pd.DataFrame(columns=df.columns)
    else:  # Return the entire DataFrame if no instruction is provided
        return df


class PandasInstructionParser(ChainableOutputParser):
    """Pandas instruction parser.

    This 'output parser' takes in pandas instructions (in Python code) and
    executes them to return an output.
    """

    def __init__(
            self, df: pd.DataFrame, output_kwargs: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Initializes parameters.

        :param pd.DataFrame df: The DataFrame to initialize with.
        :param Optional[Dict[str, Any]] output_kwargs: Additional keyword arguments (optional).
        """
        self.df = df
        self.output_kwargs = output_kwargs or {}

    def parse(self, output: str) -> Any:
        """
        Parses, validates, and corrects errors programmatically.

        :param str output: The output string to parse.
        :returns: The parsed result.
        :rtype: Any
        """
        result_df = default_output_processor(output, self.df, **self.output_kwargs)

        # To be sure take the contents from the original df but only keep rows that are in the result_df
        # (In cases where the LLM only returns a subset of the original dfs columns)
        result_df = self.df[self.df.index.isin(result_df.index)]

        return result_df


class PandasQueryRetriever(PandasQueryEngine, BaseRetriever, DataFrameRetrieverBase):
    def __init__(self, df: pd.DataFrame, df_column_description_dict: Dict[str, str] = None,
                 pandas_prompt_template: PromptTemplate = None, instruction_str: str = None,
                 instruction_parser: PandasInstructionParser = None, similarity_top_k=10, **kwargs):
        self.df_column_description_dict = df_column_description_dict or DF_COLUMN_DESCRIPTION_DICT
        self.similarity_top_k = similarity_top_k

        # Initialize the PandasQueryEngine
        super().__init__(df=df, instruction_str=instruction_str or INSTRUCTION_STR,
                         pandas_prompt=pandas_prompt_template or PANDAS_PROMPT_TEMPLATE,
                         instruction_parser=instruction_parser or PandasInstructionParser(df=df, output_kwargs={
                             'verbose': True}), **kwargs)

    def _get_table_context(self) -> str:
        """
        Gets the table context.

        :returns: The table context as a string.
        :rtype: str
        """
        columns = self._df.columns.to_list()

        # Add descriptions to the columns which do have a description in self.df_column_description_dict
        column_descriptions = [f"'{col}' ({self.df_column_description_dict.get(col, '')})"
                               if col in self.df_column_description_dict else f"'{col}'" for col in columns]

        return ', '.join(d for d in column_descriptions)

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        raise NotImplementedError

    def retrieve(self, str_or_query_bundle: QueryType) -> pd.DataFrame:
        """
        Overrides retrieve to return a DataFrame.

        :param QueryType str_or_query_bundle: Query string or QueryBundle (which optionally contains the embedding).
        :returns: Retrieved results as a DataFrame.
        :rtype: pd.DataFrame
        """
        results = super().query(str_or_query_bundle)
        result_df = results.metadata['raw_pandas_output']

        if self.similarity_top_k is not None:
            result_df = result_df.head(self.similarity_top_k)

        return result_df

    async def aretrieve(self, str_or_query_bundle: QueryType) -> pd.DataFrame:
        """
        Overrides aretrieve to return a DataFrame.

        :param QueryType str_or_query_bundle: Query string or QueryBundle (which optionally contains the embedding).
        :returns: Retrieved results as a DataFrame.
        :rtype: pd.DataFrame
        """
        results = await super().aquery(str_or_query_bundle)
        result_df = results.metadata['raw_pandas_output']

        if self.similarity_top_k is not None:
            result_df = result_df.head(self.similarity_top_k)

        return result_df
