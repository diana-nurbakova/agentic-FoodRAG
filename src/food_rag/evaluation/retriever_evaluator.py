import asyncio
import numpy as np
import os
import pandas as pd
from llama_index.core import Response
from llama_index.core import QueryBundle
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.evaluation import RetrieverEvaluator, RetrievalMetricResult
from llama_index.core.evaluation.retrieval.base import RetrievalEvalMode, RetrievalEvalResult
from llama_index.core.evaluation.retrieval.metrics import resolve_metrics
from llama_index.core.schema import QueryType
from pandas import DataFrame
from pathlib import Path
from tqdm.notebook import tqdm
from typing import Any, List, Tuple, Optional, Dict

from src.food_rag.evaluation import custom_metrics
from src.food_rag.retriever.df_retriever_base import DataFrameRetrieverBase
from src.food_rag.utils import qa_helper, hummus_docify
from src.utils import load_util

RUN_PATH = 'runs'


def summarize_results(results, key: str = None):
    """
    Summarizes the evaluation results.

    :param results: A dictionary containing as key the name of the retriever and as values a dict of evaluation metrics.
    :param key: Optional key to filter the results by a specific field, else doesn't use the filter.

    :return: A pandas DataFrame with the respective metrics as given in the results.
    """
    if key:
        results = {method: {metric: metric_results[key] for metric, metric_results in method_results.items() if
                            key in metric_results} for method, method_results in results.items()}

    return pd.DataFrame(results).T


def summarize_analyse_results(data: Dict[str, Dict[str, Any]], key: str | None = None, save_case_study: bool = True,
                              k: int = 5, case_metrics: Optional[List[str]] = None,
                              data_path: Optional[str | Path] = None, run_name: str = "",
                              normalize_scores: Optional[Dict[str, Tuple[Tuple[float, float], bool]]] = None,
                              include_meta: bool = True) -> pd.DataFrame:
    """
    Summarize and analyze evaluation results from multiple approaches.

    :param Dict[str, Dict[str, Any]] data:
        Mapping from approach name to a dict with keys
        ``"serialized_results"`` (serialized metrics) and optionally ``"responses"``
        (list of :class:`Response` objects).

    :param Optional[str] key:
        If set, extract only ``metric_results[key]`` from each metric block.

    :param bool save_case_study:
        When *True*, for every ``approach × metric`` build top-*k* and
        bottom-*k* example tables and store them via ``load_util.store_data``;
        nothing is added to the returned DataFrame.

    :param int k:
        Number of best and worst examples to include in each case-study
        table.  Defaults to ``5``.

    :param Optional[List[str]] case_metrics:
        Metrics to include in the case-study tables.  If *None*, all metrics
        found in *data* are used.

    :param Optional[Path | str] data_path:
        Base directory for storing case-study tables.  Required when
        ``save_case_study`` is *True*.

    :param str run_name:
        Sub-directory name used together with *data_path* and *RUN_PATH* when
        saving case-study tables.

    :param Optional[Dict[str, Tuple[Tuple[float, float], bool]]] normalize_scores:
        Mapping ``metric → ((min, max), increasing)``.  Adds a normalised
        column ``<metric>_norm`` scaled to **0-1**, where *1* is the best
        end of the range.

    :param bool include_meta:
        If *True* and responses are present, extract all numeric fields from
        ``Response.metadata``.  Their per-approach means are added as columns
        prefixed with ``meta_``.

    :return pd.DataFrame:
        One row per approach with the requested metrics, optional *_norm*
        columns, and optional *meta_* columns.
    """
    # 1. split the input into results and (optional) responses
    serialized_results: Dict[str, Dict[str, Any]] = {}
    responses: Dict[str, List[Response]] = {}

    for name, block in data.items():
        if isinstance(block, dict) and "serialized_results" in block:
            serialized_results[name] = block["serialized_results"]
            if "responses" in block and block["responses"] is not None:
                responses[name] = block["responses"]
        elif isinstance(block, dict):  # backward-compatibility: plain serialized dict
            serialized_results[name] = block

    # 2. Keep only the field given in key
    summary_dict = {method: {metric: metric_results[key] for metric, metric_results in method_results.items() if
                            key in metric_results} for method, method_results in serialized_results.items() if key}

    # 3. build summary frame
    summary_df = pd.DataFrame(summary_dict).T

    # 4. add normalised columns
    if normalize_scores:
        for metric, ((low, high), increasing) in normalize_scores.items():
            if metric not in summary_df.columns:
                continue

            span = high - low
            if span <= 0: # bad range → skip
                continue

            raw = summary_df[metric].astype(float)

            # scale so that low → 0 and high → 1 (or the reverse if decreasing)
            norm = (raw - low) / span if increasing else (high - raw) / span
            summary_df[f"{metric}_norm"] = norm

    # 5. add meta_* columns from responses
    if include_meta:
        for name, resp_list in responses.items():
            if name not in summary_df.index:
                continue
            meta_metrics: Dict[str, List[float]] = {}
            for response in resp_list:
                if not getattr(response, "metadata", None):
                    continue
                for meta_key, meta_value in response.metadata.items():
                    if isinstance(meta_value, (int, float)):
                        meta_metrics.setdefault(meta_key, []).append(float(meta_value))

            for meta_key, meta_values in meta_metrics.items():
                summary_df.loc[name, f"meta_{meta_key}"] = np.mean(meta_values)

    # 6. optional case-study tables
    if save_case_study:
        if not all([data_path, RUN_PATH]):
            raise ValueError("Need data_path, RUN_PATH when save_case_study=True")

        if case_metrics is None:
            case_metrics = {m for mres in serialized_results.values() for m in mres}

        for approach_name, result in serialized_results.items():
            for metric in case_metrics:
                block = result.get(metric, {})
                scored = [r for r in block.get("results", []) if r.get("score") is not None]
                if not scored:
                    continue

                top = sorted(scored, key=lambda r: r["score"], reverse=True)[:k]
                bottom = sorted(scored, key=lambda r: r["score"])[:k]

                rows = [
                    {
                        "example_type": tag,
                        "question": r["query"],
                        "response": r["response"],
                        "score": r["score"],
                        "feedback": r.get("feedback", ""),
                    }
                    for tag, part in (("Top", top), ("Bottom", bottom))
                    for r in part
                ]

                df = pd.DataFrame(rows)
                df["example_type"] = pd.Categorical(df["example_type"],["Top", "Bottom"], True)
                df = df.sort_values(["example_type", "score"],  ascending=[True, False])

                # Save the tables
                load_util.store_data(
                    data={f"{approach_name}_{metric}_case_study": df},
                    data_path=Path(data_path, RUN_PATH, run_name),
                )

    return summary_df


class DataFrameRetrieverEvaluator(RetrieverEvaluator):
    recipes_df: Optional[DataFrame] = None
    retriever: (BaseRetriever, DataFrameRetrieverBase) = None

    def evaluate(self, query: str, expected_ids: List[str], expected_texts: Optional[List[str]] = None,
                 mode: RetrievalEvalMode = RetrievalEvalMode.TEXT, **kwargs: Any, ) -> RetrievalEvalResult:
        """
        Overrides the evaluate method from BaseRetrievalEvaluator to handle both synchronous and asynchronous contexts.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # Running inside existing async context
            future = asyncio.ensure_future(self.aevaluate(
                query=query,
                expected_ids=expected_ids,
                expected_texts=expected_texts,
                mode=mode,
                **kwargs,
            ))
            return loop.run_until_complete(future)
        else:
            # Safe to use asyncio.run()
            return asyncio.run(self.aevaluate(
                query=query,
                expected_ids=expected_ids,
                expected_texts=expected_texts,
                mode=mode,
                **kwargs,
            ))

    async def aevaluate(self, query: str, expected_ids: List[str], expected_texts: Optional[List[str]] = None,
                        embedding=None, mode: RetrievalEvalMode = RetrievalEvalMode.TEXT,
                        **kwargs: Any) -> RetrievalEvalResult:
        """
        Overrides to pass on the kwargs to the metric calls and to add the recipes_df to the kwargs.

        :param str query: The input query.
        :param List[str] expected_ids: List of expected recipe IDs.
        :param Optional[List[str]] expected_texts: List of expected recipe texts (optional).
        :param RetrievalEvalMode mode: Mode of evaluation (default is RetrievalEvalMode.TEXT).
        :param embedding: The embedding of the query (optional).
        :param kwargs: Additional keyword arguments.
        :returns: Retrieval evaluation result.
        :rtype: RetrievalEvalResult
        """
        kwargs['recipes_df'] = self.recipes_df
        metric_dict = {}

        # Add the embedding if provided
        query_bundle = QueryBundle(query_str=query, embedding=embedding)

        retrieved_ids, retrieved_texts = await self._aget_retrieved_ids_and_texts(query_bundle, mode)

        # If expected IDs are provided, the score when evaluation fails is 0.0, otherwise it is nan
        if expected_ids:
            fail_score = 0.0
        else:
            fail_score = float("nan")

        for metric in self.metrics:
            try:  # for metrics with kwargs
                eval_result = metric.compute(
                    query, expected_ids, retrieved_ids, expected_texts, retrieved_texts, **kwargs
                )
            except TypeError:  # for metrics without kwargs
                try:
                    eval_result = metric.compute(
                        query, expected_ids, retrieved_ids, expected_texts, retrieved_texts
                    )
                except ValueError:
                    eval_result = RetrievalMetricResult(
                        score=fail_score,
                        metadata={"metric_name": metric.metric_name, "query": query, "expected_ids": expected_ids,
                                  "retrieved_ids": retrieved_ids, "details": "No expected or retrieved documents."},
                    )
            except ValueError:
                eval_result = RetrievalMetricResult(
                    score=fail_score,
                    metadata={"metric_name": metric.metric_name, "query": query, "expected_ids": expected_ids,
                              "retrieved_ids": retrieved_ids, "details": "No expected or retrieved documents."},
                )

            metric_dict[metric.metric_name] = eval_result

        return RetrievalEvalResult(
            query=query,
            expected_ids=[str(int_id) for int_id in expected_ids],
            expected_texts=expected_texts,
            retrieved_ids=[str(int_id) for int_id in retrieved_ids],
            retrieved_texts=retrieved_texts,
            mode=mode,
            metric_dict=metric_dict,
        )

    async def _aget_retrieved_ids_and_texts(
            self, str_or_query_bundle: QueryType, mode: RetrievalEvalMode = RetrievalEvalMode.TEXT
    ) -> Tuple[List[str], List[str]]:
        """
        Overrides to handle retrieved DataFrame instead of nodes.

        :param QueryType str_or_query_bundle: Query string or QueryBundle (which optionally contains the embedding).
        :param RetrievalEvalMode mode: The retrieval evaluation mode (default is TEXT).
        :returns: A tuple containing lists of retrieved IDs and texts.
        :rtype: Tuple[List[str], List[str]]
        """
        # Use the custom retriever to get a DataFrame
        retrieved_df = await self.retriever.aretrieve(str_or_query_bundle)

        # Extract recipe IDs and text (e.g., name and ingredients combined)
        retrieved_ids = retrieved_df['recipe_id'].tolist()
        retrieved_texts = [doc.text for doc in hummus_docify.recipes_to_docs(retrieved_df, verbose=False)]

        return retrieved_ids, retrieved_texts

    @classmethod
    def from_metric_names(cls, metric_names: List[str], **kwargs: Any) -> "DataFrameRetrieverEvaluator":
        """
        Creates evaluator from metric names.

        :param List[str] metric_names: List of metric names.
        :param kwargs: Additional arguments for the evaluator.
        :returns: A DataFrameRetrieverEvaluator instance.
        :rtype: DataFrameRetrieverEvaluator
        """
        metric_types = []
        for metric_name in metric_names:
            if isinstance(metric_name, type):
                metric_types.append(metric_name)
            else:
                try:
                    metric_type = resolve_metrics([metric_name])[0]
                    metric_types.append(metric_type)
                except (TypeError, AttributeError, KeyError, ValueError):  # seems to be a custom metric
                    try:
                        metric_type = custom_metrics.METRIC_REGISTRY[metric_name]
                        metric_types.append(metric_type)
                    except KeyError:
                        raise ValueError(
                            f"Metric {metric_name} not found in the default or the custom metric registry.")

        # Init evaluator with metrics
        return cls(metrics=[metric() for metric in metric_types], **kwargs)

    def evaluate_dataset(self, qa_set: dict, data_path, sub_dir_name, batch_size=10, show_progress: bool = True):
        """
        Evaluates the performance given retriever on a question-answer set.

        :param qa_set: Dictionary where keys are questions and values are lists of valid recipe IDs.
        :param data_path: Path to the data directory.
        :param sub_dir_name: Sub-directory name to store the evaluation results.
        :param batch_size: Size of each batch for processing.
        :param show_progress: Boolean flag to show a progress bar during evaluation (default is True).

        :return: The predicted recipes (responses of retrieval) and a dictionary containing evaluation metrics.
        """
        save_path = Path(data_path, RUN_PATH, sub_dir_name if sub_dir_name else '')

        # Load existing results
        responses, loaded_batch_names = load_existing_batches(save_path)
        processed_questions = set([response['question'] for response in responses if 'question' in response])

        # Prepare batches
        questions_to_process = [(index, qa) for index, qa in qa_set.items() if
                                qa['question'] not in processed_questions]
        batches = [questions_to_process[i:i + batch_size] for i in range(0, len(questions_to_process), batch_size)]

        iterator = tqdm(enumerate(batches), total=len(batches), disable=not show_progress, desc="Processing Batches")
        print(f"Processing {len(questions_to_process)} questions in {len(batches)} batches.")

        for batch_index, batch in iterator:
            batch_responses = self._process_batch(batch, batch_index, show_progress)

            # Append new results and save
            responses.extend(batch_responses)
            loaded_batch_names = save_batch(batch_responses, save_path, batch_index, loaded_batch_names)

        # Aggregate metrics, but ignore float("nan") values
        metric_scores = [response.get('metric_vals_dict', []) for response in responses]
        metrics = {metric_name: np.nanmean([d[metric_name] for d in metric_scores]) for metric_name in metric_scores[0]}

        return responses, metrics

    def _process_batch(self, batch, batch_index, show_progress):
        """
        Processes a single batch of questions and returns the responses and metric scores.

        :param batch: List of (question, data) tuples to process.
        :param batch_index: Index of the batch.
        :param show_progress: Boolean flag to print progress.
        :return: Tuple of (responses, metric_scores) for the batch.
        """
        batch_responses = []

        if show_progress:
            print(f"Processing batch {batch_index} with {len(batch)} questions while evaluating the retriever.")

        iterator = tqdm(batch, disable=not show_progress, desc="Processing Questions")

        for _, data in iterator:
            question = data['question']
            embedding = data.get('embedding', None)
            expected_ids = data['answer']
            expected_recipe_df = qa_helper.recipe_id_to_df(expected_ids, self.recipes_df)
            expected_texts = [
                doc.text for doc in hummus_docify.recipes_to_docs(expected_recipe_df, verbose=False)
            ]

            # Retrieve candidates and evaluate
            try:
                response = self.evaluate(
                    query=question, embedding=embedding, expected_ids=expected_ids, expected_texts=expected_texts
                )
            except Exception as e:
                print(f"Error while processing question: {question}\n{e}\nSkipping this question.")
                continue

            # Collect results
            response = {'question': question,
                        'retrieved_ids': response.retrieved_ids,
                        'metric_vals_dict': response.metric_vals_dict}
            batch_responses.append(response)

        return batch_responses


def load_existing_batches(save_path, max_batches=None):
    """
    Load previously saved batches from the save path.

    :param save_path: path to load the previously saved batches from
    :param max_batches: maximum number of batches to load. If None, all batches are loaded.

    :return: list of responses and list of loaded batch names.
    """
    responses = []
    loaded_batch_names = []
    if not os.path.exists(save_path):
        load_util.ensure_dir(save_path)
    else:
        batch_counter = 0

        # Iterate through files in the save_path directory
        for file_name in tqdm(os.listdir(save_path), desc="Loading batches"):
            if max_batches is None or batch_counter < max_batches:
                if file_name.startswith("batch_") and file_name.endswith(".pkl"):
                    full_path = os.path.join(save_path, file_name)
                    with open(full_path, 'rb') as f:
                        response = load_util.load_file(full_path)
                        response_len = len(response)
                        responses.extend(response)
                        loaded_batch_names.append(file_name)
                        batch_counter += 1
                    print(f"Loaded {file_name} with {response_len} response(s).")
                else:
                    print(f"Skipping file {file_name} as it does not match the batch naming convention.")

    return responses, loaded_batch_names


def save_batch(responses, save_path, batch_index, loaded_batch_names):
    """
    Save the current batch of responses to a file.

    :param responses: list of responses to save
    :param save_path: path where the batch is saved
    :param batch_index: index of the current batch
    :param loaded_batch_names: list of names of loaded batches

    :return: updated list of loaded batch names
    """
    file_name = f"batch_{batch_index}.pkl"

    if file_name in loaded_batch_names:
        print(f"Batch {batch_index} already present. Storing this batch with a different name.")

    while file_name in loaded_batch_names:
        batch_index += 1
        file_name = f"batch_{batch_index}.pkl"

    full_path = os.path.join(save_path, file_name)

    # Save the batch to a file
    load_util.store_file(responses, full_path)

    loaded_batch_names.append(file_name)
    print(f"Batch {batch_index} saved to {full_path}")

    return loaded_batch_names
