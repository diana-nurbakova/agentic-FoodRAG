import json
import pandas as pd

from src.food_rag.meal_plan.meal_plan_structures import MealPlanQuestions, MealPlan, MealPlanValidation, \
    MealPlanSubstitution
from src.food_rag.meal_plan.meal_type import MealType


class MealPlanEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, MealPlanQuestions) or isinstance(obj, MealPlan) or isinstance(obj, MealPlanValidation) \
                or isinstance(obj, MealPlanSubstitution) \
                or obj.__class__.__name__ == MealPlanQuestions.__name__ \
                or obj.__class__.__name__ == MealPlan.__name__ \
                or obj.__class__.__name__ == MealPlanValidation.__name__ \
                or obj.__class__.__name__ == MealPlanSubstitution.__name__:
            # Hacky solution to check if the object is a MealPlanQuestions, MealPlan, MealPlanValidation, or MealPlanSubstitution
            return obj.to_dict()
        elif isinstance(obj, MealType):
            return obj.value
        elif isinstance(obj, dict):
            return {self.default(k): self.default(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.default(i) for i in obj]
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient="records")
        else:
            return super().default(obj)
