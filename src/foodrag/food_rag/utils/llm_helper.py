import asyncio
import time
from IPython.display import Markdown, display
from llama_index.core.base.llms.types import CompletionResponse, CompletionResponseGen, CompletionResponseAsyncGen
from llama_index.core.llms.callbacks import llm_completion_callback
from llama_index.llms.litellm import LiteLLM
from typing import Any

from foodrag.food_rag.utils.nebula_helper import clean_nebula_entities as clean


def get_rag_contexts(response):
    if len(response.source_nodes) == 0:
        return response
    else:
        full_context = []

        for node in response.source_nodes:
            context = node.metadata.get('graph_store_response', node.metadata.get('kg_rel_map', node.metadata))

            if context:
                # Clean context if applicable
                cleaned_lines = _iterate_and_clean(context, clean)

                full_context.append(cleaned_lines)

    return full_context


def _iterate_and_clean(obj, clean_fn):
    if isinstance(obj, dict):
        cleaned_dict = {}

        for key, value in obj.items():
            # Clean the key
            new_key = clean_fn(key) if isinstance(key, str) else key

            # Clean the value recursively
            cleaned_dict[new_key] = _iterate_and_clean(value, clean_fn)

        return cleaned_dict
    elif isinstance(obj, list):  # Recursively clean each item in the list
        return [_iterate_and_clean(value, clean_fn) for value in obj]
    elif isinstance(obj, str):  # Clean the string
        return clean_fn(obj)
    else:  # Return other types unchanged
        return obj


def format_rag_answer(response):
    if len(response.source_nodes) == 0:
        return response
    else:
        formatted_text = ''

        for node in response.source_nodes:
            text = node.get_content()

            # Clean
            cleaned_lines = clean(text)

            # Add the joined and cleaned lines into the final formatted text
            formatted_text += '\n'.join(cleaned_lines)

        return formatted_text


def display_prompts(query_engine):
    prompts_dict = query_engine.get_prompts()

    if not prompts_dict or len(prompts_dict) == 0:
        print("No prompts available.")
    else:
        for k, p in prompts_dict.items():
            text_md = f"**Prompt Key**: {k}<br>" f"**Text:** <br>"
            display(Markdown(text_md))

            if p is not None:
                print(p.get_template())
            else:
                print("-- No prompt available --")

            display(Markdown("<br><br>"))


def serialize_and_calc_evaluation_result(eval_results):
    output = {}

    for metric_name, metric_results in eval_results.items():
        results = []
        valid_scores = []

        for r in metric_results:
            results.append({
                'contexts': r.contexts,
                'feedback': r.feedback,
                'invalid_reason': r.invalid_reason,
                'invalid_result': r.invalid_result,
                'pairwise_source': r.pairwise_source,
                'passing': r.passing,
                'query': r.query,
                'response': r.response,
                'score': r.score
            })

            if r.score is not None and isinstance(r.score, (int, float)):
                valid_scores.append(r.score)

        output[metric_name] = {
            'results': results,
            'length': len(results),
            'scored_length': len(valid_scores),
            'sum_scores': sum(valid_scores),
            'overall_score': (sum(valid_scores) / len(valid_scores)) if valid_scores else None
        }

    return output


def print_serialized_scores(serialized_results):
    for metric_name, metric_results in serialized_results.items():
        print(f"Metric: {metric_name}")
        print(f"Overall Score: {metric_results['overall_score']} ({metric_results['sum_scores']}/"
              f"{metric_results['scored_length']})")
        print(f"Number of invalid Results: {metric_results['length'] - metric_results['scored_length']}")
        print("\n")


def get_serialized_entities(response, entity_type=None):  # TODO INCLUDE TYPE
    return list(filter(None, [recipe.get('title', None) for recipe in response.metadata.values()]))


class SyncOnlyLiteLLM(LiteLLM):
    """A synchronous-only wrapper for LiteLLM."""
    delay_seconds: float = 1.0

    def __init__(self, delay_seconds=1.0, **kwargs):
        super().__init__(**kwargs)
        self.delay_seconds = delay_seconds

    @llm_completion_callback()
    def complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponse:
        """Complete a given prompt with optional formatting. Wraps the original complete method and only adds a delay
        between calls."""
        print("test_sync")
        time.sleep(self.delay_seconds)
        return super().complete(prompt, formatted=formatted, **kwargs)

    @llm_completion_callback()
    def stream_complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponseGen:
        """Stream completion for a given prompt with optional formatting. Wraps the original stream_complete method."""
        print("test_sync_stream")
        time.sleep(self.delay_seconds)
        return super().stream_complete(prompt, formatted=formatted, **kwargs)

    @llm_completion_callback()
    async def acomplete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponse:
        """Disables asynchronous completion and runs the synchronous complete method."""
        asyncio.run(asyncio.sleep(self.delay_seconds))
        return asyncio.run(super().acomplete(prompt, formatted, **kwargs))

    @llm_completion_callback()
    async def astream_complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponseAsyncGen:
        """Disables asynchronous streaming completion and runs the synchronous stream_complete method."""
        asyncio.run(asyncio.sleep(self.delay_seconds))
        return asyncio.run(super().astream_complete(prompt, formatted, **kwargs))
