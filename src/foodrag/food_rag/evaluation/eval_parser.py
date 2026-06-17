import json

import asyncio
import logging
import pandas as pd
import re
from llama_index.core import PromptTemplate
from llama_index.core.base.response.schema import Response
from llama_index.core.evaluation import BatchEvalRunner
from llama_index.core.evaluation.base import BaseEvaluator, EvaluationResult
from llama_index.core.output_parsers import PydanticOutputParser
from llama_index.core.output_parsers.utils import extract_json_str
from llama_index.core.schema import TextNode, NodeWithScore
from llama_index.core.types import Model
from pathlib import Path
from tenacity import retry, stop_after_attempt, wait_exponential
from typing import Any, Dict, Optional, Tuple, List, cast, Type

from foodrag.food_rag.utils import hummus_docify
from foodrag.utils import load_util

# Most prompts are directly copied from the original files, but some have been modified to fit the recipe context.
# Refer to llama_index/core/evaluation/correctness.py, llama_index/core/evaluation/guideline.py,
# llama_index/core/evaluation/relevancy.py, and llama_index/core/evaluation/faithfulness.py for the original prompts.
CORRECTNESS_EVAL_TEMPLATE = PromptTemplate("""
You are an expert evaluation system for a recipe recommender.

You are given the following information:
- a user query, and
- a generated answer, and
- a context including valid recipe titles.

Your job is to judge the relevance and correctness of the generated answer.
Output a single score that represents a holistic evaluation.
You must return your response in a line with only the score.
Do not return answers in any other format.
On a separate line provide your reasoning for the score as well.

Regarding the nutri score, the who score, and the fsa score: Those are between 0 and 1, where 0 is the worst and 1 is 
the best. So, i.e., if the query asks for a recipe with a nutri score of 0.5, the answer should contain a recipe with a 
nutri score of 0.5 or higher.

Follow these guidelines for scoring:
- Your score has to be between 1 and 5, where 1 is the worst and 5 is the best.
- If the generated answer is not relevant to the user query, you should give a score of 1.
- If the generated answer is relevant but contains mistakes, you should give a score between 2 and 3.
- If the generated answer is relevant and fully correct, you should give a score between 4 and 5.

Example Response:
4.0
The generated answer has the exact same metrics as the reference answer, but it is not as concise.

## User Query
{query}

## Generated Answer
{generated_answer}

## Context
{reference_answer}
""")

GUIDELINE_EVAL_TEMPLATE = PromptTemplate("""
Here is the original query:
Query: {query}
Critique the following response based solely on the guidelines below:
Response: {response}
Guidelines: {guidelines}
Now please provide constructive criticism.
""")

RELEVANCY_EVAL_TEMPLATE = PromptTemplate("""
Your task is to evaluate if the recipe or meal plan recommendation for the query and response is in line with the 
context information provided. For that, simply check if the recipe title of the response is present in the context and 
is relevant to the query.
You have two options to answer. Either YES/ NO.
Answer - YES, if the response for the query is in line with context information otherwise NO.
Query and Response/recommendation:
{query_str}
Context:
{context_str}
Answer:
""")

FAITHFULNESS_EVAL_TEMPLATE = PromptTemplate("""
Please tell if a given recipe or meal plan recommendation is supported by the context. For that, simply check if the 
recipe title is present in the context. If no recipe title is present in the context, the information/recommendation is 
only valid if it is also empty.
Only check the context for the recipe titles, not the content and not the nutrition of the recipes.
You need to answer with either YES or NO.
Answer YES if any of the context supports the information, even if most of the context is unrelated. Some examples are 
provided below.

----Examples start----
Information/recommendation: Banana Bread Oatmeal
Context: Valid recipes (by titles): Banana-Oat Bran Muffins, Appetizer Grape Jelly and Chili Sauce Meatballs or 
Lil Smokies, Lite or Light Pumpkin Chocolate Chip Muffins, Kate's Afternoon Wine Cooler (Zwt - France), Banana Bread 
Oatmeal, Baked Banana Stuffed French Toast
Answer: YES

Information/recommendation: Banana Bread Oatmeal for vegans
Context: Valid recipes (by titles): Banana-Oat Bran Muffins, Appetizer Grape Jelly and Chili Sauce Meatballs or 
Lil Smokies, Lite or Light Pumpkin Chocolate Chip Muffins, Kate's Afternoon Wine Cooler (Zwt - France), Banana Bread 
Oatmeal, Baked Banana Stuffed French Toast
Answer: NO

Information/recommendation: "Recipe Id":2, "Title": "Banana Bread Oatmeal", "Description": "I love this recipe.", "Nutri Score": 0.5, 
"FSA Score": 0.5, "WHO Score": 0.5, "Recipe Type": "Breakfast", "Cuisine": "American".
Context: Valid recipes (by titles): Banana-Oat Bran Muffins, Appetizer Grape Jelly and Chili Sauce Meatballs or 
Lil Smokies, Lite or Light Pumpkin Chocolate Chip Muffins, Kate's Afternoon Wine Cooler (Zwt - France), Banana Bread 
Oatmeal, Baked Banana Stuffed French Toast
Answer: YES
----Examples end----


Information/recommendation: {query_str}
Context: {context_str}
Answer: 
""")


def score_reasoning_parser(eval_response: str) -> Tuple[Optional[float], Optional[str]]:
    """
    Parser function for evaluation response. This function extracts the score and reasoning from a llm response. It is
    an extended version of the default parser found in llama_index/core/evaluation_eval_utils.py.

    Args:
        eval_response (str): The response string from the evaluation.

    Returns:
        Tuple[float, str]: A tuple containing the score as a float and the reasoning as a string.
    """
    if not eval_response.strip():
        # Return None or default values if the response is empty
        return None, "No response"

    # Remove the <think></think> tags and the content which is in between (more precise: the content before the last
    # enclosing think tag)
    if "</think>" in eval_response:
        eval_response = eval_response.split("</think>")[-1].strip()

    # Split the response into score and reasoning
    try:
        score_str, reasoning_str = eval_response.split("\n", 1)
    except Exception:
        score_str = None
        reasoning_str = eval_response

    try:
        score = float(score_str)
    except (ValueError, TypeError):
        score = None

    reasoning = reasoning_str.lstrip("\n")
    return score, reasoning


class ContextBatchEvalRunner(BatchEvalRunner):
    """ContextBatchEvalRunner.

    This class is used to run the evaluation on a batch of contexts.
    It is functionally similar to the `BatchEvalRunner` class, but inserts context information into responses source
    nodes if not already provided.
    """

    def __init__(self, evaluators: Dict[str, BaseEvaluator], save_path: str,
                 file_name: str = "eval_runner_temp.pkl",
                 show_progress: bool = False, batch_size: int = 1, logging_level: int = logging.INFO,
                 counter_logging_level: int = logging.INFO, context_single_mode: bool = True) -> None:
        """
        Initialize the ContextBatchEvalRunner.

        Args:
            evaluators (Dict[str, BaseEvaluator]): Dictionary of evaluators.
            save_path (str): Path to save the evaluation results.
            file_name (str): Name of the file to save the evaluation results.
            show_progress (bool): Whether to show progress bars. Defaults to False.
            batch_size (int): The size of the batch to process at once. Defaults to 1.
            logging_level (int): The logging level to use for the evaluation. Defaults to logging.INFO.
            counter_logging_level (int): The logging level to use for the counter. Defaults to logging.INFO.
            context_single_mode (bool): If True, only one context is used for each response. Else, splits the context
             into multiple nodes which are summarized by the evaluators. Defaults to True. Note: Only relevant if the
                context is a string. If the context is a dataframe, it will always be split into multiple nodes.
        """
        super().__init__(evaluators=evaluators, workers=1, show_progress=show_progress)
        self.save_path = Path(save_path)
        self.file_name = file_name
        self.batch_size = batch_size
        self.counter_logging_level = counter_logging_level
        self.context_single_mode = context_single_mode

        logging.basicConfig(level=logging_level)

    @retry(
        reraise=True,
        stop=stop_after_attempt(10),
        wait=wait_exponential(multiplier=1, min=4, max=10),
    )
    async def aeval_response_worker(
            self,
            semaphore: asyncio.Semaphore,
            evaluator: BaseEvaluator,
            evaluator_name: str,
            query: Optional[str] = None,
            response: Optional[Response] = None,
            eval_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, EvaluationResult]:
        """Get aevaluate_response tasks with semaphore."""
        eval_kwargs = eval_kwargs or {}

        eval_kwargs_wo_context = self.inject_context(response, eval_kwargs)

        async with semaphore:
            try:
                eval_result = await evaluator.aevaluate_response(
                    query=query, response=response, **eval_kwargs_wo_context
                )
            except ValueError as e:  # An value error might happen if available llm context size is too small for the to-be
                # evaluated prompt. E.g.: \llama_index\core\indices\prompt_helper.py:153,
                # PromptHelper._get_available_context_size(self, num_prompt_tokens):
                # ValueError: Calculated available context size -290 was not non-negative.
                # ==> Retry without the context param as it might be too large.
                # Else, return a result with passing=False.
                try:
                    # remove context by removing the source nodes
                    response.response = self.shorten_recipe_forgiving(response.response, max_len=500)
                    eval_result = await evaluator.aevaluate_response(
                        query=query, response=response, **eval_kwargs_wo_context
                    )
                except ValueError as e:
                    try:
                        # remove context by removing the source nodes
                        response.response = self.shorten_recipe_forgiving(response.response, max_len=100)
                        eval_result = await evaluator.aevaluate_response(
                            query=query, response=response, **eval_kwargs_wo_context
                        )
                    except ValueError as e:
                        eval_result = EvaluationResult(
                            query=query,
                            response=response.response,
                            passing=False,
                            score=None,  # None score will be interpreted as invalid and not counted in the overall score
                            feedback=f"The response is invalid or the context is too small. Error:\n{e}",
                        )

            return evaluator_name, eval_result

    @retry(
        reraise=True,
        stop=stop_after_attempt(10),
        wait=wait_exponential(multiplier=1, min=4, max=10),
    )
    def eval_response_worker(
            self,
            evaluator: BaseEvaluator,
            query: Optional[str] = None,
            response: Optional[Response] = None,
            eval_kwargs: Optional[Dict[str, Any]] = None,
    ) -> EvaluationResult:
        """Evaluate a single query/response with optional context injection."""
        eval_kwargs = eval_kwargs or {}

        eval_kwargs_wo_context = self.inject_context(response, eval_kwargs)

        try:
            eval_result = evaluator.evaluate_response(
                query=query, response=response, **eval_kwargs_wo_context
            )
        except ValueError as e:  # An value error might happen if available llm context size is too small for the to-be
            # evaluated prompt. E.g.: \llama_index\core\indices\prompt_helper.py:153,
            # PromptHelper._get_available_context_size(self, num_prompt_tokens):
            # ValueError: Calculated available context size -290 was not non-negative.
            # ==> Retry without the context param as it might be too large.
            # Else, return a result with passing=False.
            try:
                # remove context by removing the source nodes
                response.response = self.shorten_recipe_forgiving(response.response, max_len=500)
                eval_result = evaluator.evaluate_response(
                    query=query, response=response, **eval_kwargs_wo_context
                )
            except ValueError as e:
                try:
                    # remove context by removing the source nodes
                    response.response = self.shorten_recipe_forgiving(response.response, max_len=100)
                    eval_result = evaluator.evaluate_response(
                        query=query, response=response, **eval_kwargs_wo_context
                    )
                except ValueError as e:
                    eval_result = EvaluationResult(
                        query=query,
                        response=response.response,
                        passing=False,
                        score=None,  # None score will be interpreted as invalid and not counted in the overall score
                        feedback=f"The response is invalid or the context is too small. Error:\n{e}",
                    )

        return eval_result

    def inject_context(self, response, eval_kwargs):
        if 'contexts' in eval_kwargs:
            context = eval_kwargs['contexts']

            if isinstance(context, str):  # Context is only string of valid recipe titles
                if self.context_single_mode:
                    response.source_nodes = [NodeWithScore(node=TextNode(text=context))]
                else:
                    try:
                        context = context.split('----------------------')[1].strip()
                    except IndexError:
                        pass
                    response.source_nodes = [NodeWithScore(node=TextNode(text=context_line)) for
                                             context_line in context.split("\n")]
            elif isinstance(context, pd.DataFrame):  # Context is dataframe with valid recipes
                response.source_nodes = hummus_docify.recipes_to_docs(context, add_type=False, verbose=False)

            # remove context nodes from kwargs
            eval_kwargs_wo_context = eval_kwargs.copy()
            del eval_kwargs_wo_context['contexts']
        else:
            eval_kwargs_wo_context = eval_kwargs
        return eval_kwargs_wo_context

    @staticmethod
    def shorten_recipe_forgiving(raw_str, max_len=100) -> str:
        # Try to find the JSON object in the string using a regex
        json_match = re.search(r'\{.*\}', raw_str, re.DOTALL)
        if not json_match:
            # Fall back: use basic text substitution
            raw_str = re.sub(r'\s+', ' ', raw_str).strip()
            return raw_str[:max_len] + ('...' if len(raw_str) > max_len else '')

        json_str = json_match.group()

        try:
            recipe = json.loads(json_str)
        except json.JSONDecodeError:
            # Try to fix common JSON issues like trailing commas
            fixed_str = re.sub(r',\s*([}\]])', r'\1', json_str)  # remove trailing commas
            fixed_str = re.sub(r'\\r\\n', ' ', fixed_str)        # replace escape linebreaks
            try:
                recipe = json.loads(fixed_str)
            except Exception:
                # Final fallback: treat as plain text
                flat_text = re.sub(r'\s+', ' ', raw_str).strip()
                return flat_text[:max_len] + ('...' if len(flat_text) > max_len else '')

        def trim(text):
            if not isinstance(text, str):
                text = str(text)
            text = re.sub(r'\s+', ' ', text).strip()
            return text[:max_len] + ('...' if len(text) > max_len else '')

        for key in ["description", "directions", "tags"]:
            if key in recipe:
                recipe[key] = trim(recipe[key])

        # Fallback if key not present: inject placeholder
        for key in ["description", "directions", "tags"]:
            if key not in recipe:
                recipe[key] = "N/A"

        try:
            return json.dumps(recipe, ensure_ascii=False, indent=2)
        except Exception:
            # Fallback: flatten dict to string
            flat = str(recipe)
            return flat[:max_len] + ('...' if len(flat) > max_len else '')

    async def aevaluate_responses(
            self,
            queries: Optional[List[str]] = None,
            responses: Optional[List[Response]] = None,
            **eval_kwargs_lists: Dict[str, Any],
    ) -> Dict[str, List[EvaluationResult]]:
        raise NotImplementedError("This method is not implemented in the ContextBatchEvalRunner. Use "
                                  "`evaluate_responses` instead.")

    def evaluate_responses_async(
            self,
            queries: Optional[List[str]] = None,
            responses: Optional[List[Response]] = None,
            **eval_kwargs_lists: Dict[str, Any],
    ) -> Dict[str, List[EvaluationResult]]:
        """
        Evaluate query, response pairs.

        This evaluates queries and response objects.

        Args:
            queries (Optional[List[str]]): List of query strings. Defaults to None.
            responses (Optional[List[Response]]): List of response objects.
                Defaults to None.
            **eval_kwargs_lists (Dict[str, Any]): Dict of either dicts or lists
                of kwargs to pass to evaluator. Defaults to None.
                    multiple evaluators: {evaluator: {kwarg: [list of values]},...}
                    single evaluator:    {kwarg: [list of values]}

        """
        queries, responses = self._validate_and_clean_inputs(queries, responses)
        eval_kwargs_lists = self._validate_nested_eval_kwargs_types(eval_kwargs_lists)

        # boolean to check if using multi kwarg evaluator
        multi_kwargs = len(eval_kwargs_lists) > 0 and isinstance(
            next(iter(eval_kwargs_lists.values())), dict
        )

        # Load already computed evaluations from the file
        results: Dict[str, List[EvaluationResult]] = self.update_and_persist_evaluation(self.save_path, self.file_name)

        # run evaluations
        eval_jobs = []
        num_loaded_evals = 0
        for idx, query in enumerate(cast(List[str], queries)):
            response = cast(List, responses)[idx]
            for name, evaluator in self.evaluators.items():
                # Skip all already computed evaluations from the queries and responses
                evaluator_results = results.get(name, [])
                matching = ContextBatchEvalRunner._matching_eval(evaluator_results, query, response.response)
                if matching is not None:
                    # If the evaluation already exists, skip it
                    num_loaded_evals += 1
                    continue

                if multi_kwargs:
                    # multi-evaluator - get appropriate runtime kwargs if present
                    kwargs = (
                        eval_kwargs_lists[name] if name in eval_kwargs_lists else {}
                    )
                else:
                    # single evaluator (maintain backwards compatibility)
                    kwargs = eval_kwargs_lists

                eval_kwargs = self._get_eval_kwargs(kwargs, idx)
                eval_jobs.append(
                    self.aeval_response_worker(
                        self.semaphore,
                        evaluator,
                        name,
                        query=query,
                        response=response,
                        eval_kwargs=eval_kwargs,
                    )
                )

        # Log how many evaluations were loaded from the file
        logging.log(level=self.counter_logging_level,
                    msg=f"Loaded {num_loaded_evals} evaluations from the file {self.file_name}.")

        total_batches = (len(eval_jobs) + self.batch_size - 1) // self.batch_size
        for batch_idx, i in enumerate(range(0, len(eval_jobs), self.batch_size), start=1):
            batch = eval_jobs[i:i + self.batch_size]
            batch_results = self.run_async(self.asyncio_mod.gather(*batch))
            formatted_results: Dict[str, List[EvaluationResult]] = self._format_results(batch_results)

            for evaluator_name, eval_results in formatted_results.items():
                if evaluator_name not in results:
                    results[evaluator_name] = []
                # Append new results to the existing ones
                results[evaluator_name].extend(eval_results)

            # Store batch
            results = self.update_and_persist_evaluation(self.save_path, self.file_name, results)

            # Print batch number, size, and total number of batches
            logging.log(level=self.counter_logging_level,
                        msg=f"Processed batch {batch_idx} of size {len(batch)} of {total_batches} total batches.")

        return results

    def evaluate_responses(
            self,
            queries: Optional[List[str]] = None,
            responses: Optional[List[Response]] = None,
            **eval_kwargs_lists: Dict[str, Any],
    ) -> Dict[str, List[EvaluationResult]]:
        """
        Evaluate query, response pairs.

        Args:
            queries (Optional[List[str]]): List of query strings.
            responses (Optional[List[Response]]): List of response objects.
            **eval_kwargs_lists (Dict[str, Any]): Kwargs for evaluators.

        Returns:
            Dict[str, List[EvaluationResult]]: Evaluation results.
        """
        queries, responses = self._validate_and_clean_inputs(queries, responses)
        eval_kwargs_lists = self._validate_nested_eval_kwargs_types(eval_kwargs_lists)

        # boolean to check if using multi kwarg evaluator
        multi_kwargs = len(eval_kwargs_lists) > 0 and isinstance(
            next(iter(eval_kwargs_lists.values())), dict
        )

        # Load already computed evaluations from the file
        results: Dict[str, List[EvaluationResult]] = self.update_and_persist_evaluation(self.save_path, self.file_name)
        num_loaded_evals = 0

        for idx, query in enumerate(queries):
            response = responses[idx]
            for name, evaluator in self.evaluators.items():
                # Skip all already computed evaluations from the queries and responses
                existing = results.get(name, [])
                if self._matching_eval(existing, query, response.response):  # If the evaluation already exists, skip it
                    num_loaded_evals += 1
                    print(
                        f"Skipping evaluation for query-response pair number {idx} for evaluator '{name}' as it already exists.")
                    continue

                kwargs = eval_kwargs_lists.get(name, {}) if multi_kwargs else eval_kwargs_lists
                eval_kwargs = self._get_eval_kwargs(kwargs, idx)

                eval_result = self.eval_response_worker(
                    evaluator, query=query, response=response, eval_kwargs=eval_kwargs
                )
                results.setdefault(name, []).append(eval_result)

                # Persist after each evaluation
                results = self.update_and_persist_evaluation(self.save_path, self.file_name, results)

            # Log the evaluations
            print(f"Evaluated query-response pair number {idx} for all evaluators.")

        print(f"Loaded {num_loaded_evals} evaluations from the file {self.file_name}.")

        return results

    @staticmethod
    def update_and_persist_evaluation(save_path: Path, file_name: str = "eval_runner_temp.pkl",
                                      evaluations: Dict[str, List[EvaluationResult]] = None) \
            -> Dict[str, List[EvaluationResult]]:
        """
        Appends the current question and its response to a file.

        :param save_path: Path of the dict.
        :param file_name: Name of the file to save the responses.
        :param evaluations: The evaluations of a certain evaluator.

        returns: The loaded and present evaluations combined.
        """
        full_path = save_path / file_name

        # Load the existing responses from the file
        if full_path.exists():
            if file_name.endswith('.json'): # re-convert to EvaluationResult from json serializable format
                loaded_evaluations: Dict[str, List[Any]] = load_util.load_file(full_path)
                loaded_evaluations: Dict[str, List[EvaluationResult]] = {
                    evaluator_name: [EvaluationResult.model_validate(eval_result) for eval_result in evals]
                    for evaluator_name, evals in loaded_evaluations.items()
                }
            else:
                loaded_evaluations: Dict[str, List[EvaluationResult]] = load_util.load_file(full_path)

        else:
            loaded_evaluations: Dict[str, List[EvaluationResult]] = {}

        # Append all not existing responses
        if evaluations is not None:
            for evaluator_name, eval_list in evaluations.items():
                for evaluation in eval_list:
                    question = evaluation.query
                    response = evaluation.response
                    loaded = loaded_evaluations.get(evaluator_name, [])

                    matching = ContextBatchEvalRunner._matching_eval(loaded, question, response)

                    if matching is None:
                        if evaluator_name not in loaded_evaluations:
                            loaded_evaluations[evaluator_name] = []

                        loaded_evaluations[evaluator_name].append(evaluation)


        if file_name.endswith('.json'): # convert EvaluationResult to make it json serializable
            serializable_evaluations = {
                evaluator_name: [eval_result.model_dump() for eval_result in evals]
                for evaluator_name, evals in loaded_evaluations.items()
            }
        else:
            serializable_evaluations = loaded_evaluations

        # Save to file
        load_util.store_file(serializable_evaluations, full_path,
                             force_suffix='.json' if file_name.endswith('.json') else None)

        return loaded_evaluations

    @staticmethod
    def _matching_eval(evals: List[EvaluationResult], question: str, response: str) -> EvaluationResult | None:
        matching = None
        for load in evals:
            if load.query == question and load.response == response:
                matching = load
                break
        return matching

    @staticmethod
    def run_async(coro):
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            future = asyncio.ensure_future(coro)
            return loop.run_until_complete(future)
        else:
            return asyncio.run(coro)


class CorrectingPydanticOutputParser(PydanticOutputParser):
    """PydanticOutputParser for CorrectingEvaluationData."""

    def __init__(self, output_default: Model, output_cls: Type[Model], **kwargs: Any) -> None:
        """
        Init params. Example to create a CorrectingPydanticOutputParser:
        output_parser = CorrectingPydanticOutputParser(
            output_default=EvaluationData(
                feedback="",
                passing=False,
            ),
            output_cls=EvaluationData
        )

        :param output_default: Default output object to use if parsing fails. The parser will the default object and
         try to fill in the missing values from the parsed JSON string.
        :param output_cls: Pydantic output class.
        :param kwargs: Additional arguments to pass to the parent class.
        """
        super().__init__(output_cls=output_cls, **kwargs)
        self.output_default = output_default

    def parse(self, text: str) -> Any:
        """Parse the text like the super-parser but if it fails tries to correct it otherwise."""
        json_str = extract_json_str(text)

        try:
            result = self._output_cls.model_validate_json(json_str)
        except Exception:
            # Try to get to find an inner structure
            extract_left = json_str.find("{")
            extract_right = json_str.rfind("}")
            json_str_extract = json_str[extract_left: extract_right + 1].strip()

            try:
                result = self._output_cls.model_validate_json(json_str_extract)
            except Exception:
                # Try to get to find an inner structure #2
                json_str_extract = json_str[extract_left + 1: extract_right].strip()  # remove the outer brackets

                extract_extract_left = json_str_extract.find("{")
                extract_extract_right = json_str_extract.rfind("}")
                json_str_extract_extract = json_str_extract[extract_extract_left: extract_extract_right + 1].strip()

                try:
                    result = self._output_cls.model_validate_json(json_str_extract_extract)
                except Exception:
                    # If all fails, return the default object and try to fill in the missing values
                    result = self.output_default

                    # aggregate all from the json extracted field to the default if possible
                    for key, value in self.output_default.model_dump().items():
                        if key in json_str:
                            try:
                                extracted_value = extract_values_from_string(json_str, key)

                                if len(extracted_value) > 0:
                                    result[key] = extracted_value[0]  # just take the first occurrence
                            except Exception:
                                pass
        return result


def extract_values_from_string(data: str, key: str) -> List[Any]:
    """
    Extracts all values associated with the given key from a JSON-like string,
    without parsing the string as JSON.

    :param str data: The JSON-like string.
    :param str key: The key to search for.
    :return: A list of extracted values.
    :rtype: list
    """
    # Match keys like "key": true, "key": "some text", "key": 123
    pattern = rf'"{re.escape(key)}"\s*:\s*(true|false|null|"[^"]*"|\d+(?:\.\d+)?)'
    matches = re.findall(pattern, data)

    # Clean string quotes if needed
    cleaned = []
    for match in matches:
        if match.startswith('"') and match.endswith('"'):
            cleaned.append(match[1:-1])  # Strip quotes
        elif match == "true":
            cleaned.append(True)
        elif match == "false":
            cleaned.append(False)
        elif match == "null":
            cleaned.append(None)
        elif re.match(r'^\d+(\.\d+)?$', match):
            cleaned.append(float(match) if '.' in match else int(match))
        else:
            cleaned.append(match)

    return cleaned
