from enum import Enum


class MealType(Enum):
    """An enumeration of meal types."""
    BREAKFAST = "breakfast"
    LUNCH = "lunch"
    DINNER = "dinner"
    SNACK = "snack"

    @staticmethod
    def get_all():
        return [MealType.BREAKFAST, MealType.LUNCH, MealType.DINNER, MealType.SNACK]

    @staticmethod
    def get_all_str():
        return [meal_type.value for meal_type in MealType.get_all()]

    def __str__(self):
        return self.value

    def __repr__(self):
        return self.value

    def __eq__(self, other):
        if isinstance(other, str):
            return self.value == other.lower()
        elif isinstance(other, MealType):
            return self.value == other.value
        elif isinstance(other, Enum) and self.__class__.__name__ == other.__class__.__name__:
            # Somehow, maybe via different import strategies, the other MealType is not recognized as such.
            # Side effect: Same-named but other enums with same values might be evaluated as equal.
            return self.value == other.name.lower()
        return False

    def __hash__(self):
        return hash(self.value)
