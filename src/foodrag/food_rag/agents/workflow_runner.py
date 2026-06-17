import asyncio
import nest_asyncio
from llama_index.core import Response
from llama_index.core.agent.workflow import AgentWorkflow, AgentStream, AgentInput, AgentOutput, ToolCallResult, \
    ToolCall
from llama_index.core.memory.types import BaseChatStoreMemory
from llama_index.core.workflow import step, Workflow
from llama_index.core.workflow.context import Context
from llama_index.core.workflow.events import StopEvent, StartEvent
from pathlib import Path
from tqdm.notebook import tqdm
from typing import List, Dict, Optional, Any

from foodrag.food_rag.tools.tool_base import RecipeToolBase
from foodrag.food_rag.meal_plan.meal_plan_structures import MealPlanQuestions, MealPlanValidation, MealPlan, \
    MealPlanSubstitution
from foodrag.food_rag.utils.recipe_context import RecipeState
from foodrag.utils import load_util

nest_asyncio.apply()  # Required to run the notebook in Jupyter for disabling async operations in the notebook
RUN_PATH = 'runs'


def run_all_questions_on_tool(questions: List[str], tool: RecipeToolBase, data_path, sub_dir_name,
                              file_name: str = "workflow_responses_temp.pkl",
                              store_each_response=True, verbose=True, **kwargs) \
        -> Dict[str, str] | Dict[str, Dict[str, Any]]:
    """Run given tool for all questions.
    :param questions: List of questions to run the tool on.
    :param tool: Tool to run.
    :param data_path: Path to the data directory.
    :param sub_dir_name: Sub-directory name to store the evaluation results.
    :param file_name: Name of the file to save the responses.
    :param store_each_response: Whether to write each response on the disk, or only once for all.
    :param verbose: Whether to print the responses.
    :param kwargs: Additional arguments to pass to the tool.
    """
    save_path = Path(data_path, RUN_PATH, sub_dir_name if sub_dir_name else '')

    # Create the directory if it does not exist
    load_util.ensure_dir(save_path)

    # Load the existing responses from the file, else it creates a new one
    question_and_responses: Dict[str, str] | Dict[str, Dict[str, Any]] = update_and_persist_response(
        save_path, file_name, verbose=verbose)

    # Run the tool for each question
    for idx, question in tqdm(enumerate(questions), disable=not verbose, desc="Running Tool on Questions"):
        if question in question_and_responses:
            # Skip the question if it already exists
            if verbose:
                print(f"Question #{idx} (of total {len(questions)}) already in the file.")
        else:
            # Run the tool
            kwargs.update({"question": question})
            ctx = get_dummy_state().get_context()
            tool_output = tool.to_tool().call(ctx=ctx, **kwargs)

            # Get current state from the context
            try:
                current_state: RecipeState = RecipeState.from_context(ctx)
            except Exception:
                current_state = None

            # Get response
            try:
                response_str = tool_output.content
            except Exception:
                response_str = "Error: No response for the question."

            # Add the response to the question and responses dictionary
            question_and_responses[question] = {'response': response_str}

            # Get metadata if available
            if current_state is not None:
                try:
                    metadata = {}

                    llm_parsing_usage_counter = current_state.get_llm_parsing_usage_counter()
                    llm_usage_counter = current_state.get_llm_usage_counter()
                    llm_sequential_usage_counter = current_state.get_llm_sequential_usage_counter()

                    if llm_parsing_usage_counter is not None:
                        metadata[RecipeState.LLM_PARSING_USAGE_COUNTER_KEY] = llm_parsing_usage_counter
                    if llm_usage_counter is not None:
                        metadata[RecipeState.LLM_USAGE_COUNTER_KEY] = llm_usage_counter
                    if llm_sequential_usage_counter is not None:
                        metadata[RecipeState.LLM_SEQUENTIAL_USAGE_COUNTER_KEY] = llm_sequential_usage_counter

                    if len(metadata) == 0:
                        metadata = None
                except Exception:
                    metadata = None

                # Add the metadata if available
                if metadata is not None:
                    question_and_responses[question]['metadata'] = metadata

            # Store the response
            if store_each_response:
                update_and_persist_response(save_path, file_name, question_and_responses, idx, verbose=verbose)

    # Store all responses (only necessary if store_each_response is False as otherwise they are already stored)
    if not store_each_response:
        update_and_persist_response(save_path, file_name, question_and_responses, verbose=verbose)

    return question_and_responses


def run_all_questions(questions: List[str], agent_workflow: AgentWorkflow, data_path, sub_dir_name,
                      file_name: str = "workflow_responses_temp.pkl",
                      memory: Optional[BaseChatStoreMemory] = None, store_each_response=True, verbose=True) \
        -> Dict[str, str]:
    """Run the workflow for all questions. Reset the memory for each question.

    :param questions: List of questions to run the workflow on.
    :param memory: Memory object to store the state.
    :param agent_workflow: Agent workflow to run.
    :param data_path: Path to the data directory.
    :param sub_dir_name: Sub-directory name to store the evaluation results.
    :param file_name: Name of the file to save the responses.
    :param store_each_response: Whether to write each response on the disk, or only once for all.
    :param verbose: Whether to print the responses.
    """
    return asyncio.run(
        arun_all_questions(questions, agent_workflow, data_path, sub_dir_name, file_name, memory, store_each_response,
                           verbose))


async def arun_all_questions(questions: List[str], agent_workflow: AgentWorkflow, data_path, sub_dir_name,
                             file_name: str = "workflow_responses_temp.pkl",
                             memory: Optional[BaseChatStoreMemory] = None, store_each_response=True, verbose=True,
                             do_print_workflow=True) -> Dict[str, str]:
    """Run the workflow for all questions. Reset the memory for each question.

    :param questions: List of questions to run the workflow on.
    :param memory: Memory object to store the state.
    :param agent_workflow: Agent workflow to run.
    :param data_path: Path to the data directory.
    :param sub_dir_name: Sub-directory name to store the evaluation results.
    :param file_name: Name of the file to save the responses.
    :param store_each_response: Whether to write each response on the disk, or only once for all.
    :param verbose: Whether to print the responses.
    :param do_print_workflow: Whether to print the workflow.
    """
    save_path = Path(data_path, RUN_PATH, sub_dir_name if sub_dir_name else '')

    # Create the directory if it does not exist
    load_util.ensure_dir(save_path)

    # Load the existing responses from the file, else it creates a new one
    question_and_responses = update_and_persist_response(save_path, file_name, verbose=verbose)

    # Run the workflow for each question
    for idx, question in tqdm(enumerate(questions), disable=not verbose, desc="Running Workflow on Questions"):
        if question in question_and_responses:
            # Skip the question if it already exists
            if verbose:
                print(f"Question #{idx} (of total {len(questions)}) already in the file.")
        else:
            if memory:
                # Reset the memory for each question
                memory.reset()

                # Run the workflow
                if do_print_workflow:
                    handler = agent_workflow.run(user_msg=question, memory=memory)
                    await print_workflow(handler)
                else:
                    handler = await agent_workflow.run(user_msg=question, memory=memory)
            else:
                # Run the workflow without memory
                if do_print_workflow:
                    handler = agent_workflow.run(user_msg=question)
                    await print_workflow(handler)
                else:
                    handler = await agent_workflow.run(user_msg=question)

            # Get response
            response = await handler.ctx.get("state")  # TODO CONVERT TO REAL RESPONSE OBJECT
            question_and_responses[question] = response

            # Store the response
            if store_each_response:
                update_and_persist_response(save_path, file_name, question_and_responses, idx, verbose=verbose)

    # Store all responses (only necessary if store_each_response is False as otherwise they are already stored)
    if not store_each_response:
        update_and_persist_response(save_path, file_name, question_and_responses, verbose=verbose)

    return question_and_responses


def update_and_persist_response(save_path: Path, file_name: str = "workflow_responses_temp.pkl",
                                question_and_responses: Dict[str, str] | Dict[str, Dict[str, Any]] = None,
                                response_index=None, verbose=True) -> Dict[str, str] | Dict[str, Dict[str, Any]]:
    """
    Appends the current question and its response to a file.

    :param save_path: Path of the dict.
    :param file_name: Name of the file to save the responses.
    :param question_and_responses: The questions and responses to update and persist.
    :param response_index: The index of the response which is used for the printing (if enabled), if None, the index is
     not printed.
    :param verbose: Whether to print the response index.

    returns: The full question and response dict.
    """
    full_path = save_path / file_name

    # Load the existing responses from the file
    if full_path.exists():
        loaded_question_and_responses: Dict[str, str] | Dict[str, Dict[str, Any]] = load_util.load_file(full_path)
    else:
        loaded_question_and_responses: Dict[str, Dict[str, str]] = dict(dict())

    # Append all not existing responses
    if question_and_responses is not None:
        for question, response in question_and_responses.items():
            if question not in loaded_question_and_responses:
                loaded_question_and_responses[question] = response

    # Save to file
    load_util.store_file(loaded_question_and_responses, full_path)

    # Print the response if response_index is provided
    if response_index is not None:
        if verbose:
            print(f"Saved response #{response_index}.")

    return loaded_question_and_responses


async def print_workflow(handler):
    current_agent = None
    async for event in handler.stream_events():
        if (
                hasattr(event, "current_agent_name")
                and event.current_agent_name != current_agent
        ):
            current_agent = event.current_agent_name
            print(f"\n{'=' * 50}")
            print(f"🤖 Agent: {current_agent}")
            print(f"{'=' * 50}\n")

        if isinstance(event, AgentStream):
            if event.delta:
                print(event.delta, end="", flush=True)
        elif isinstance(event, AgentInput):
            print("📥 Input:", event.input)
        elif isinstance(event, AgentOutput):
            if event.response.content:
                print("📤 Output:", event.response.content)
            if event.tool_calls:
                print(
                    "🛠️  Planning to use tools:",
                    [call.tool_name for call in event.tool_calls],
                )
        elif isinstance(event, ToolCallResult):
            print(f"🔧 Tool Result ({event.tool_name}):")
            print(f"  Arguments: {event.tool_kwargs}")
            print(f"  Output: {event.tool_output}")
        elif isinstance(event, ToolCall):
            print(f"🔨 Calling Tool: {event.tool_name}")
            print(f"  With arguments: {event.tool_kwargs}")
        elif isinstance(event, StopEvent):
            print("🛑 Stop Event")
        else:
            print(event)


def get_dummy_state(state_data: Dict = {}) -> RecipeState:
    """Create a dummy context for testing purposes and set the initial state.

    :param state_data: The additional state data to set in the context. Overwrites corresponding key value pairs in the
     initial state. Keeps key value pairs of the initial state if they are not present in this parameter.
    :return: The RecipeState object.
    """

    class DummyStopEvent(StopEvent):
        pass

    class DummyWorkflow(Workflow):
        @step()
        def start(self, event: StartEvent) -> DummyStopEvent:
            yield DummyStopEvent()

    # Instantiate the dummy workflow and dummy context
    ctx = Context(workflow=DummyWorkflow())

    # set init data
    state = RecipeState(ctx)

    # set given data
    for key, value in state_data.items():
        if key == RecipeState.MEAL_PLAN_QUESTIONS_KEY:
            value = MealPlanQuestions.from_dict(value)
        elif key == RecipeState.MEAL_PLAN_RECOMMENDATION_KEY:
            value = MealPlan.from_dict(value)
        elif key == RecipeState.MEAL_PLAN_VALIDATION_KEY:
            value = MealPlanValidation.from_dict(value)
        elif key == RecipeState.SUBSTITUTED_MEAL_PLAN_RECOMMENDATION_KEY:
            value = MealPlanSubstitution.from_dict(value)
        state[key] = value
    state.save_to_context()
    return state


def to_ordered_response(questions: List[str], question_and_responses: Dict[str, str] | Dict[str, Dict[str, Any]]) \
        -> List[Response]:
    """
    Convert the question and responses to a list of Response objects with the same order as the questions.

    :param questions: The questions whose order should be used. Also, only those questions are considered.
    :param question_and_responses: The question and responses to convert. If there are questions in this dict which are
    not in the questions list, they are ignored.
    :return: The list of Response objects.
    """
    ordered: List[Response] = []

    for question in questions:
        if question in question_and_responses:
            response = question_and_responses[question]

            if isinstance(response, dict):  # question_and_responses are Dict[str, Dict[str, Any]]
                if 'response' in response:
                    answer = response['response']
                else:
                    answer = None

                if 'metadata' in response:
                    metadata = response['metadata']

                    if not isinstance(metadata, dict):
                        metadata = {"metadata": metadata}  # Ensure metadata is a dict
                else:
                    metadata = {}

                if answer is not None:
                    ordered.append(Response(response=answer, metadata=metadata))
                else:
                    ordered.append(Response(response="Error: No response for the question.", metadata=metadata))
            elif isinstance(response, str):  # question_and_responses are Dict[str, str]
                ordered.append(Response(response=response))
            else:
                ordered.append(Response(response="Error: Invalid response format."))
        else:
            ordered.append(Response(response="Error: No response for the question."))

    return ordered
