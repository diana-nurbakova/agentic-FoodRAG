# Source: https://docs.llamaindex.ai/en/stable/examples/agent/agent_runner/agent_runner_rag_controllable/


def chat_repl(agent, exit_when_done: bool = True):
    """Chat REPL.

    :param RecipeAgentBase agent: The agent to chat with, e.g. ReActAgent.
    :param bool exit_when_done: if True, automatically exit when step is finished.
            Set to False if you want to keep going even if step is marked as finished by the agent.
            If False, you need to explicitly call "exit" to finalize a task execution.

    """
    task_message = None
    while task_message != "exit":
        task_message = input(">> Human: ")
        if task_message == "exit":
            break

        task = agent.create_task(task_message)

        response = None
        step_output = None
        message = None
        # TODO INTEGRATE chat_history: Optional[List[ChatMessage]] = ...
        while message != "exit":
            if message is None or message == "":
                step_output = agent.run_step(task.task_id)
            else:
                step_output = agent.run_step(task.task_id, input=message)
            if exit_when_done and step_output.is_last:
                print(
                    ">> Task marked as finished by the agent, executing task execution."
                )
                break

            message = input(
                ">> Add feedback during step? (press enter/leave blank to continue, and type 'exit' to stop): "
            )
            if message == "exit":
                break

        if step_output is None:
            print(">> You haven't run the agent. Task is discarded.")
        elif not step_output.is_last:
            print(">> The agent hasn't finished yet. Task is discarded.")
        else:
            response = agent.finalize_response(task.task_id)
        print(f"Agent: {str(response)}")
