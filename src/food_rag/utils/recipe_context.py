import asyncio
import json
import pandas as pd
from copy import deepcopy
from llama_index.core.workflow import Context
from typing import Dict, List, Any

from food_rag.meal_plan.json_encoder import MealPlanEncoder
from src.food_rag.meal_plan.meal_plan_structures import MealPlanQuestions, MealPlan, MealPlanValidation, \
    MealPlanSubstitution
from src.food_rag.meal_plan.meal_type import MealType


class RecipeState:
    """This is the state class of the context for all the recipe tools."""
    ORIGINAL_USER_QUESTION_KEY = "original_user_question"
    MEAL_PLAN_QUESTIONS_KEY = "meal_plan_questions"
    MEAL_PLAN_RECOMMENDATION_KEY = "meal_plan_recommendation"
    OVERALL_MEAL_PLAN_VALIDATION_KEY = "overall_meal_plan_validation"
    MEAL_PLAN_VALIDATION_KEY = "meal_plan_validation"
    SUBSTITUTED_MEAL_PLAN_RECOMMENDATION_KEY = "substituted_meal_plan_recommendation"
    HISTORY_KEY = "history"
    MEAL_PLAN_EXPLANATION_KEY = "meal_plan_explanation"
    FINAL_RECOMMENDATION_KEY = "final_recommendation"
    LLM_PARSING_USAGE_COUNTER_KEY = "llm_parsing_usage_counter"
    LLM_USAGE_COUNTER_KEY = "llm_usage_counter"
    LLM_SEQUENTIAL_USAGE_COUNTER_KEY = "llm_sequential_usage_counter"

    def __init__(self, ctx: Context):
        self._data = self.get_init_state()
        self._ctx = ctx

    @staticmethod
    def get_init_state() -> Dict[str, Any]:
        """
        Get the initial state of the recipe context.

        :return: The initial state.
        """
        return {
            RecipeState.ORIGINAL_USER_QUESTION_KEY: "No recorded question yet.",  # the original user question
            RecipeState.MEAL_PLAN_QUESTIONS_KEY: MealPlanQuestions(),  # a dict meal_type -> question
            RecipeState.MEAL_PLAN_RECOMMENDATION_KEY: MealPlan(),  # a dict meal_type -> recipe
            RecipeState.OVERALL_MEAL_PLAN_VALIDATION_KEY: "No validation yet.",  # overall validation str of the meal plan
            RecipeState.MEAL_PLAN_VALIDATION_KEY: MealPlanValidation(),  # a dict meal_type -> validation str
            RecipeState.SUBSTITUTED_MEAL_PLAN_RECOMMENDATION_KEY: MealPlanSubstitution(),  # a dict meal_type -> subs. m.p. str
            RecipeState.HISTORY_KEY: [],
            RecipeState.MEAL_PLAN_EXPLANATION_KEY: "Not written yet.",
            RecipeState.FINAL_RECOMMENDATION_KEY: "Final recommendation required.",
            RecipeState.LLM_PARSING_USAGE_COUNTER_KEY: 0,  # Counter for LLM parsing only usage
            RecipeState.LLM_USAGE_COUNTER_KEY: 0,  # Counter for any LLM usage (including parsing)
            RecipeState.LLM_SEQUENTIAL_USAGE_COUNTER_KEY: 0,
            # Counter for sequential LLM usage (including parsing if used sequential)
        }

    @staticmethod
    def from_context(ctx: Context) -> 'RecipeState':
        """
        Load the state from the context.

        :param ctx: The context to load the state from.
        :return: The loaded state.
        """
        state = RecipeState(ctx)
        state._data = asyncio.run(ctx.get("state"))[0]
        return state

    def reload_from_context(self) -> None:
        """
        Reload the state from the context.
        """
        self._data = asyncio.run(self._ctx.get("state"))[0]

    def save_to_context(self) -> None:
        """
        Save the state to the context.
        """
        asyncio.run(self._ctx.set("state", [self._data]))

    # --- Generic dict-like methods ---
    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self._data[key] = value

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def keys(self):
        return self._data.keys()

    def values(self):
        return self._data.values()

    def items(self):
        return self._data.items()

    def get(self, key: str, default=None):
        return self._data.get(key, default)

    def to_dict(self) -> Dict[str, Any]:
        return deepcopy(self._data)

    def update(self, updates: Dict[str, Any]) -> None:
        for key, value in updates.items():
            self._data[key] = value

    # --- Getters ---

    def get_original_user_question(self) -> str:
        return self._data[self.ORIGINAL_USER_QUESTION_KEY]

    def get_meal_plan_questions(self) -> MealPlanQuestions:
        return self._data[self.MEAL_PLAN_QUESTIONS_KEY]

    def get_meal_plan_recommendation(self) -> MealPlan:
        return self._data[self.MEAL_PLAN_RECOMMENDATION_KEY]

    def get_meal_plan_recommendation_dfs(self) -> Dict[MealType, pd.DataFrame]:
        meal_plan_dfs = {}
        for meal_type, recommendation_dict in self.get_meal_plan_recommendation().items():
            meal_plan_dfs[meal_type] = pd.DataFrame(recommendation_dict)
        return meal_plan_dfs

    def get_overall_meal_plan_validation(self) -> str:
        return self._data[self.OVERALL_MEAL_PLAN_VALIDATION_KEY]

    def get_meal_plan_validation(self) -> MealPlanValidation:
        return self._data[self.MEAL_PLAN_VALIDATION_KEY]

    def get_substituted_meal_plan_recommendation(self) -> MealPlanSubstitution:
        return self._data[self.SUBSTITUTED_MEAL_PLAN_RECOMMENDATION_KEY]

    def get_history(self) -> List[Any]:
        return self._data[self.HISTORY_KEY]

    def get_meal_plan_explanation(self) -> str:
        return self._data[self.MEAL_PLAN_EXPLANATION_KEY]

    def get_final_recommendation(self) -> str:
        return self._data[self.FINAL_RECOMMENDATION_KEY]

    def get_llm_usage_counter(self) -> int:
        return self._data[self.LLM_USAGE_COUNTER_KEY]

    def get_llm_sequential_usage_counter(self) -> int:
        return self._data[self.LLM_SEQUENTIAL_USAGE_COUNTER_KEY]

    def get_llm_parsing_usage_counter(self) -> int:
        return self._data[self.LLM_PARSING_USAGE_COUNTER_KEY]

    def get_context(self) -> Context:
        return self._ctx

    # --- Setters ---
    def set_original_user_question(self, question: str):
        self._data[self.ORIGINAL_USER_QUESTION_KEY] = question

    def set_meal_plan_questions(self, question: MealPlanQuestions):
        self._data[self.MEAL_PLAN_QUESTIONS_KEY] = question

    def set_meal_plan_recommendation(self, plan: MealPlan):
        self._data[self.MEAL_PLAN_RECOMMENDATION_KEY] = plan

    def set_overall_meal_plan_validation(self, validation: str):
        self._data[self.OVERALL_MEAL_PLAN_VALIDATION_KEY] = validation

    def set_meal_plan_validation(self, validation: MealPlanValidation):
        self._data[self.MEAL_PLAN_VALIDATION_KEY] = validation

    def set_substituted_meal_plan_recommendation(self, plan: MealPlanSubstitution):
        self._data[self.SUBSTITUTED_MEAL_PLAN_RECOMMENDATION_KEY] = plan

    def set_history(self, history: List[Any]):
        self._data[self.HISTORY_KEY] = history

    def set_meal_plan_explanation(self, explanation: str):
        self._data[self.MEAL_PLAN_EXPLANATION_KEY] = explanation

    def set_final_recommendation(self, recommendation: str):
        self._data[self.FINAL_RECOMMENDATION_KEY] = recommendation

    def increment_llm_usage_counter(self):
        self._data[self.LLM_USAGE_COUNTER_KEY] += 1

    def increment_llm_sequential_usage_counter(self):
        self._data[self.LLM_SEQUENTIAL_USAGE_COUNTER_KEY] += 1

    # --- Update Methods ---
    def update_meal_plan_questions(self, updates: MealPlanQuestions):
        self._data[self.MEAL_PLAN_QUESTIONS_KEY].update(updates)

    def update_meal_plan_questions_item(self, meal_type: MealType, question: str):
        self._data[self.MEAL_PLAN_QUESTIONS_KEY].update({meal_type: question})

    def update_meal_plan_recommendation(self, updates: MealPlan):
        self._data[self.MEAL_PLAN_RECOMMENDATION_KEY].update(updates)

    def update_meal_plan_recommendation_item(self, meal_type: MealType, meal_plan: str):
        self._data[self.MEAL_PLAN_RECOMMENDATION_KEY].update({meal_type: meal_plan})

    def update_meal_plan_validation(self, updates: MealPlanValidation):
        self._data[self.MEAL_PLAN_VALIDATION_KEY].update(updates)

    def update_meal_plan_validation_item(self, meal_type: MealType, validation: str):
        self._data[self.MEAL_PLAN_VALIDATION_KEY].update({meal_type: validation})

    def update_substituted_meal_plan_recommendation(self, updates: MealPlanSubstitution):
        self._data[self.SUBSTITUTED_MEAL_PLAN_RECOMMENDATION_KEY].update(updates)

    def update_substituted_meal_plan_recommendation_item(self, meal_type: MealType, meal_plan: str):
        self._data[self.SUBSTITUTED_MEAL_PLAN_RECOMMENDATION_KEY].update({meal_type: meal_plan})

    def update_history(self, updates: List[str]):
        self._data[self.HISTORY_KEY].extend(updates)

    def update_history_item(self, entry: str):
        self._data[self.HISTORY_KEY].append(entry)

    def increment_llm_counter(self, parsing_inc: int = 0, sequential_inc: int = 1, total_inc: int = 1) -> None:
        """Increment the LLM usage counter.

        :param parsing_inc: Increment for parsing usage. Defaults to 0.
        :param sequential_inc: Increment for sequential usage. Defaults to 1.
        :param total_inc: Increment for total usage. Defaults to 1.
        """
        self._data[self.LLM_PARSING_USAGE_COUNTER_KEY] += parsing_inc
        self._data[self.LLM_SEQUENTIAL_USAGE_COUNTER_KEY] += sequential_inc
        self._data[self.LLM_USAGE_COUNTER_KEY] += total_inc

    # --- Print Methods ---
    def print(self, keys: List[str] = None) -> None:
        """Print the state from the context.

        :param keys: The keys to print. If None, all keys are printed. Defaults to None.
        """
        state = self.to_dict()

        if keys is not None:
            state = {key: state[key] for key in keys if key in state}

        print(json.dumps(state, indent=4, cls=MealPlanEncoder))
