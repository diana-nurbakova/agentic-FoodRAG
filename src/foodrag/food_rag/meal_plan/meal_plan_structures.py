import pandas as pd
from pydantic import BaseModel, model_validator
from typing import Dict, Any, List
from copy import deepcopy as dpc

from foodrag.food_rag.meal_plan.meal_type import MealType
from foodrag.food_rag.utils import hummus_docify


class MealPlanQuestions(BaseModel):
    questions: Dict[MealType, str] = {}

    @model_validator(mode='before')
    def convert_keys_to_enum(cls, values):
        if "question" in values:
            values["question"] = {
                MealType(k) if isinstance(k, str) else k: v
                for k, v in values["question"].items()
            }
        return values

    def to_dict(self):
        """
        Converts the MealPlanQuestion to a dictionary with string keys.
        """
        return {meal_type.value: question for meal_type, question in self.items()}

    @classmethod
    def from_dict(cls, meal_plan_query_dict: Dict[str, str] | Dict[MealType, str]) -> 'MealPlanQuestions':
        """
        Builds the MealPlanQuestion, converting string keys to MealType where possible.
        """
        query_map = {}
        for meal_type_str, query in meal_plan_query_dict.items():
            if isinstance(meal_type_str, MealType):
                meal_type = meal_type_str
            else:
                try:
                    meal_type = MealType(meal_type_str.lower())
                except ValueError:
                    continue
            query_map[meal_type] = query
        return cls(questions=query_map)

    def __str__(self) -> str:
        """
        Returns a human-readable list of meal-type-to-question entries.
        """
        lines = [f"'{meal_type}': '{query}'" for meal_type, query in self.items()]
        return "\n".join(lines)

    def __getitem__(self, key: MealType) -> str:
        return self.questions[key]

    def __setitem__(self, key: MealType, value: str) -> None:
        self.questions[key] = value

    def items(self):
        return self.questions.items()

    def keys(self):
        return self.questions.keys()

    def values(self):
        return self.questions.values()

    def update(self, updates: 'MealPlanQuestions'):
        self.questions.update(updates)

    def get(self, key: MealType, default=None) -> str:
        return self.questions.get(key, default)

    def __iter__(self):
        return iter(self.questions)

    def __len__(self):
        return len(self.questions)

    def __contains__(self, key):
        return key in self.questions

    def __eq__(self, other):
        return self.questions == other.query

    def __ne__(self, other):
        return self.questions != other.query

    def deepcopy(self) -> 'MealPlanQuestions':
        """
        Creates a deep copy of the MealPlanQuestions instance.
        """
        return MealPlanQuestions(questions=dpc(self.questions))


class MealPlan(BaseModel):
    meal_plan: Dict[MealType, Dict[str, Any]] = {}  # meal_type -> recipe as dict (colum_name -> recipe_value)

    @model_validator(mode='before')
    def convert_keys_to_enum(cls, values):
        if "meal_plan" in values:
            values["meal_plan"] = {
                MealType(k) if isinstance(k, str) else k: v
                for k, v in values["meal_plan"].items()
            }
        return values

    def to_dict(self):
        """
        Converts the MealPlan to a dictionary with string keys.
        """
        return {meal_type.value: recipe for meal_type, recipe in self.items()}

    @classmethod
    def from_dict(cls, meal_plan_dict: Dict[str, Dict[str, Any]] | Dict[MealType, Dict[str, Any]]) -> 'MealPlan':
        """
        Builds the MealPlan, converting string keys to MealType where possible.
        """
        meal_plan = {}
        for meal_type_str, recipe_dict in meal_plan_dict.items():
            if isinstance(meal_type_str, MealType):
                meal_type = meal_type_str
            else:
                try:
                    meal_type = MealType(meal_type_str.lower())
                except ValueError:
                    continue
            meal_plan[meal_type] = recipe_dict
        return cls(meal_plan=meal_plan)

    def __str__(self) -> str:
        """
        Returns a human-readable list of meal-type-to-recipe entries.
        """
        lines = [f"'{meal_type}':\n{recipe}" for meal_type, recipe in self.items()]
        return "\n------------\n".join(lines)

    def get_short_str(self) -> str:
        """
        Returns a human-readable list of meal-type-to-recipe-title entries.
        """
        lines = [f"'{meal_type}': '{recipe.get('title', 'no valid recipe found')}'"
                 for meal_type, recipe in self.items()]
        return "\n".join(lines)

    def get_recommendation_str(self) -> str:
        """
        Returns a human-readable meal plan recommendation.
        """
        return self.get_partial_recommendation_str()

    def get_partial_recommendation_str(self, meal_types: List[MealType] = None, features: List[str] = None) -> str:
        """
        Returns a human-readable meal plan recommendation.

        :param meal_types: List of meal types to include in the recommendation. Default is None, which includes all
        present meal types.
        :param features: List of features to include in the recommendation. Default is None, which includes all
        present features.
        """
        if meal_types is None:
            meal_types = list(self.keys())

        def get_formatted_recipe(recipe: Dict[str, Any], features: List[str] = None):
            # Keep only the features that are in the recipe and in the features list
            if features is not None:
                recipe = {k: recipe[k] for k in features if k in recipe}

            # Convert the recipe to a string
            doc_list = hummus_docify.recipes_to_docs(pd.DataFrame.from_dict(recipe, orient='index').T
                                                     if len(recipe) > 0 else pd.DataFrame(),
                                                     add_type=False, verbose=False)
            if len(doc_list) == 0:
                return None
            else:
                # Get the first document and convert it to a string
                return doc_list[0].text

        lines = [
            f"Recommendation for {meal_type}:\n'{formatted}'"
            for meal_type in meal_types
            if (formatted := get_formatted_recipe(recipe=self[meal_type], features=features)) is not None
        ]

        if not lines:
            lines = ["Sorry, there are no recipes to recommend!"]

        return "\n\n".join(lines)

    def __getitem__(self, key: MealType) -> Dict[str, Any]:
        return self.meal_plan[key]

    def __setitem__(self, key: MealType, value: Dict[str, Any]):
        self.meal_plan[key] = value

    def items(self):
        return self.meal_plan.items()

    def keys(self):
        return self.meal_plan.keys()

    def values(self):
        return self.meal_plan.values()

    def update(self, updates: 'MealPlan'):
        self.meal_plan.update(updates)

    def __iter__(self):
        return iter(self.meal_plan)

    def __len__(self):
        return len(self.meal_plan)

    def __contains__(self, key):
        return key in self.meal_plan

    def __eq__(self, other):
        return self.meal_plan == other.meal_plan

    def __ne__(self, other):
        return self.meal_plan != other.meal_plan

    def deepcopy(self) -> 'MealPlan':
        """
        Creates a deep copy of the MealPlan instance.
        """
        return MealPlan(meal_plan=dpc(self.meal_plan))


class MealPlanValidation(BaseModel):
    validation: Dict[MealType, str] = {}

    @model_validator(mode='before')
    def convert_keys_to_enum(cls, values):
        if "validation" in values:
            values["validation"] = {
                MealType(k) if isinstance(k, str) else k: v
                for k, v in values["validation"].items()
            }
        return values

    def to_dict(self) -> Dict[str, str]:
        """
        Converts the validation dict to a string-keyed dictionary.
        """
        return {meal_type.value: result for meal_type, result in self.items()}

    @classmethod
    def from_dict(cls, validation_dict: Dict[str, str] | Dict[MealType, str]) -> 'MealPlanValidation':
        """
        Builds MealPlanValidation from a string-keyed dictionary.
        """
        validation_map = {}
        for meal_type_str, result in validation_dict.items():
            if isinstance(meal_type_str, MealType):
                meal_type = meal_type_str
            else:
                try:
                    meal_type = MealType(meal_type_str.lower())
                except ValueError:
                    continue
            validation_map[meal_type] = result
        return cls(validation=validation_map)

    def __getitem__(self, key: MealType) -> str:
        return self.validation[key]

    def __setitem__(self, key: MealType, value: str):
        self.validation[key] = value

    def items(self):
        return self.validation.items()

    def keys(self):
        return self.validation.keys()

    def values(self):
        return self.validation.values()

    def update(self, updates: 'MealPlanValidation'):
        self.validation.update(updates)

    def get(self, key: MealType, default=None) -> str:
        return self.validation.get(key, default)

    def __iter__(self):
        return iter(self.validation)

    def __len__(self):
        return len(self.validation)

    def __contains__(self, key):
        return key in self.validation

    def __eq__(self, other):
        return self.validation == other.validation

    def __ne__(self, other):
        return self.validation != other.validation

    def __str__(self) -> str:
        """
        Returns a human-readable list of meal-type-to-validation entries.
        """
        lines = [f"'{meal_type}':\n'{result}'" for meal_type, result in self.items()]
        return "\n".join(lines)

    def get_short_str(self) -> str:
        """
        Returns a human-readable list of meal-type-to-validation entries.
        """
        lines = [f"'{meal_type}': '{self.shorten_validation_text(result)}'" for meal_type, result in self.items()]
        return "\n".join(lines)

    @staticmethod
    def shorten_validation_text(validation_text: str) -> str:
        """
        Shortens the validation text to a more concise version.
        """
        return "VALID:".join(validation_text.replace('\n', '').replace('\r', '').split('VALID'))

    def deepcopy(self) -> 'MealPlanValidation':
        """
        Creates a deep copy of the MealPlanValidation instance.
        """
        return MealPlanValidation(validation=dpc(self.validation))


class MealPlanSubstitution(BaseModel):
    substitution: Dict[MealType, str] = {}

    @model_validator(mode='before')
    def convert_keys_to_enum(cls, values):
        if "substitution" in values:
            values["substitution"] = {
                MealType(k) if isinstance(k, str) else k: v
                for k, v in values["substitution"].items()
            }
        return values

    def to_dict(self) -> Dict[str, str]:
        """
        Converts the substitution dict to a string-keyed dictionary.
        """
        return {meal_type.value: result for meal_type, result in self.items()}

    @classmethod
    def from_dict(cls, substitution_dict: Dict[str, str] | Dict[MealType, str]) -> 'MealPlanSubstitution':
        """
        Builds MealPlanSubstitution from a string-keyed dictionary.
        """
        substitution_map = {}
        for meal_type_str, result in substitution_dict.items():
            if isinstance(meal_type_str, MealType):
                meal_type = meal_type_str
            else:
                try:
                    meal_type = MealType(meal_type_str.lower())
                except ValueError:
                    continue
            substitution_map[meal_type] = result
        return cls(substitution=substitution_map)

    def __getitem__(self, key: MealType) -> str:
        return self.substitution[key]

    def __setitem__(self, key: MealType, value: str):
        self.substitution[key] = value

    def items(self):
        return self.substitution.items()

    def keys(self):
        return self.substitution.keys()

    def values(self):
        return self.substitution.values()

    def update(self, updates: 'MealPlanSubstitution'):
        self.substitution.update(updates)

    def get(self, key: MealType, default=None) -> str:
        return self.substitution.get(key, default)

    def __iter__(self):
        return iter(self.substitution)

    def __len__(self):
        return len(self.substitution)

    def __contains__(self, key):
        return key in self.substitution

    def __eq__(self, other):
        return self.substitution == other.substitution

    def __ne__(self, other):
        return self.substitution != other.substitution

    def __str__(self) -> str:
        """
        Returns a human-readable list of meal-type-to-substitution entries.
        """
        lines = [f"'{meal_type}':\n'{result}'" for meal_type, result in self.items()]
        return "\n".join(lines)

    def get_short_str(self) -> str:
        """
        Returns a human-readable list of meal-type-to-substitution entries.
        """
        lines = [f"'{meal_type}': '{result}'" for meal_type, result in self.items()]
        return "\n".join(lines)

    def is_empty(self):
        """
        Check if the substitution is empty.
        """
        return len(self.substitution.values()) == 0 or all(not result for result in self.substitution.values())

    def deepcopy(self) -> 'MealPlanSubstitution':
        """
        Creates a deep copy of the MealPlanValidation instance.
        """
        return MealPlanSubstitution(substitution=dpc(self.substitution))

