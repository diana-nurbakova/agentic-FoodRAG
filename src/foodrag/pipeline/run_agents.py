"""Stage 6: run the agentic meal-plan pipeline over a set of questions.

The agent workflow assembly (retriever + query engine + the Recommend/Correct/
Enhance agents and their tools) is involved and lives in
``notebooks/food_RAG_agents.ipynb``; pass the assembled ``AgentWorkflow`` in as
``agent_workflow``. This stage drives it over the questions and persists each
response (resumable via workflow_runner's temp file).

    run_agents.run(questions, agent_workflow, data_path="data", sub_dir_name="run_1")
"""
from __future__ import annotations

from typing import Any, Optional


def run(
    questions: list[str],
    agent_workflow: Any,
    *,
    data_path: str = "data",
    sub_dir_name: str = "agent_run",
    file_name: str = "workflow_responses_temp.pkl",
    memory: Any = None,
    store_each_response: bool = True,
    verbose: bool = True,
) -> dict:
    """Run the agent workflow over ``questions``; return the responses dict.

    ``agent_workflow`` is injectable (and required) — it is the assembled
    AgentWorkflow. Delegates to ``workflow_runner.run_all_questions``, which
    checkpoints each response so a crashed run can resume.
    """
    if agent_workflow is None:
        raise ValueError(
            "agent_workflow is required. Assemble it as in "
            "notebooks/food_RAG_agents.ipynb and pass it in."
        )
    from foodrag.food_rag.agents import workflow_runner

    return workflow_runner.run_all_questions(
        questions, agent_workflow, data_path, sub_dir_name,
        file_name=file_name, memory=memory,
        store_each_response=store_each_response, verbose=verbose,
    )
