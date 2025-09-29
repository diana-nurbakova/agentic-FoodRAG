import numpy as np
from llama_index.core.evaluation.retrieval.metrics_base import BaseRetrievalMetric, RetrievalMetricResult
from typing import Any, List, Optional


def meal_plan_score(meal_plan, actual_nutrition, k=0.01):  # TODO DEBUG
    total_penalty = 0
    total_items = 0

    # Loop through each meal in the meal plan
    for meal_type, meal_data in meal_plan.items():
        guidelines = meal_data['guidelines']
        actual_values = actual_nutrition[meal_type]

        # Loop through each nutrient in the guidelines
        for nutrient, guideline_value in guidelines.items():
            actual_value = actual_values.get(nutrient, 0)  # Handle missing nutrients in actual values

            # Calculate the deviation (relative to the guideline)
            deviation = abs(actual_value - guideline_value) / guideline_value

            # Exponential penalty for deviations
            penalty = np.exp(k * deviation) - 1

            # Accumulate the total penalty and number of nutrients considered
            total_penalty += penalty
            total_items += 1

    # Normalize the penalty to a score between 0 and 1
    max_penalty = total_items  # The worst case is 1 penalty per nutrient
    score = 1 - total_penalty / max_penalty  # Higher score is better

    # Ensure the score is within the range [0, 1]
    return max(0, min(1, score))


def avg_nutrition_score(recipes_df, score_names):
    if not recipes_df.empty:
        mean_scores = recipes_df[score_names].mean(axis=0)

        # Return the mean_scores in a dict
        return {score_name: mean_scores[score_name] for score_name in score_names}
    else:
        return {}


class WHOScoreMetric(BaseRetrievalMetric):
    """Compute the who score. Invalid scores are set to NaN (if there are no expected recipes)."""

    metric_name = "who_score"

    def compute(
            self,
            query: Optional[str] = None,
            expected_ids: Optional[List[str]] = None,
            retrieved_ids: Optional[List[str]] = None,
            expected_texts: Optional[List[str]] = None,
            retrieved_texts: Optional[List[str]] = None,
            **kwargs: Any,
    ) -> RetrievalMetricResult:
        if "recipes_df" in kwargs and not kwargs["recipes_df"].empty:
            recipes_df = kwargs["recipes_df"]

            retrieved_recipes = recipes_df[recipes_df["recipe_id"].isin(retrieved_ids)]
            who_score = retrieved_recipes[["who_score"]].mean(axis=0)

            return RetrievalMetricResult(
                score=who_score,
                metadata={"metric_name": self.metric_name,
                          "retrieved_ids": retrieved_ids},
            )
        else:
            raise KeyError("recipes_df not found in kwargs or empty.")


class FSAScoreMetric(BaseRetrievalMetric):
    """Compute the fsa score. Invalid scores are set to NaN (if there are no expected recipes)."""

    metric_name = "fsa_score"

    def compute(
            self,
            query: Optional[str] = None,
            expected_ids: Optional[List[str]] = None,
            retrieved_ids: Optional[List[str]] = None,
            expected_texts: Optional[List[str]] = None,
            retrieved_texts: Optional[List[str]] = None,
            **kwargs: Any,
    ) -> RetrievalMetricResult:
        if "recipes_df" in kwargs and not kwargs["recipes_df"].empty:
            recipes_df = kwargs["recipes_df"]

            retrieved_recipes = recipes_df[recipes_df["recipe_id"].isin(retrieved_ids)]
            fsa_score = retrieved_recipes[["fsa_score"]].mean(axis=0)

            return RetrievalMetricResult(
                score=fsa_score,
                metadata={"metric_name": self.metric_name,
                          "retrieved_ids": retrieved_ids},
            )
        else:
            raise KeyError("recipes_df not found in kwargs or empty.")


class NUTRIScoreMetric(BaseRetrievalMetric):
    """Compute the NUTRIScore. Invalid scores are set to NaN (if there are no expected recipes)."""

    metric_name = "nutri_score"

    def compute(
            self,
            query: Optional[str] = None,
            expected_ids: Optional[List[str]] = None,
            retrieved_ids: Optional[List[str]] = None,
            expected_texts: Optional[List[str]] = None,
            retrieved_texts: Optional[List[str]] = None,
            **kwargs: Any,
    ) -> RetrievalMetricResult:
        if "recipes_df" in kwargs and not kwargs["recipes_df"].empty:
            recipes_df = kwargs["recipes_df"]

            retrieved_recipes = recipes_df[recipes_df["recipe_id"].isin(retrieved_ids)]
            nutri_score = retrieved_recipes[["nutri_score"]].mean(axis=0)

            return RetrievalMetricResult(
                score=nutri_score,
                metadata={"metric_name": self.metric_name,
                          "retrieved_ids": retrieved_ids},
            )
        else:
            raise KeyError("recipes_df not found in kwargs or empty.")


class WHODeltaScoreMetric(BaseRetrievalMetric):
    """Compute the who score difference between the expected recipes and the retrieved recipes. The score is between 0
     and infinity. A score of 1 means that the retrieved score is the same as the expected one. A score below 1 means
     that the retrieved score is given factor smaller than the expected one. A score higher 1 means that the retrieved
     score is given factor greater than the expected one. Invalid scores are set to NaN (if there are no expected
     recipes)."""

    metric_name = "who_delta_score"

    def compute(
            self,
            query: Optional[str] = None,
            expected_ids: Optional[List[str]] = None,
            retrieved_ids: Optional[List[str]] = None,
            expected_texts: Optional[List[str]] = None,
            retrieved_texts: Optional[List[str]] = None,
            **kwargs: Any,
    ) -> RetrievalMetricResult:
        if "recipes_df" in kwargs and not kwargs["recipes_df"].empty:
            recipes_df = kwargs["recipes_df"]

            expected_recipes = recipes_df[recipes_df["recipe_id"].isin(expected_ids)]
            retrieved_recipes = recipes_df[recipes_df["recipe_id"].isin(retrieved_ids)]

            who_score_expected = expected_recipes[["who_score"]].mean(axis=0).get(0, np.nan)
            who_score_retrieved = retrieved_recipes[["who_score"]].mean(axis=0).get(0, np.nan)

            # Difference between the expected and retrieved who scores
            if who_score_expected == 0:
                score_difference = np.nan
            else:
                score_difference = who_score_retrieved / who_score_expected

            return RetrievalMetricResult(
                score=score_difference,
                metadata={"metric_name": self.metric_name,
                          "expected_ids": expected_ids,
                          "retrieved_ids": retrieved_ids},
            )
        else:
            raise KeyError("recipes_df not found in kwargs or empty.")


class FSADeltaScoreMetric(BaseRetrievalMetric):
    """Compute the fsa score difference between the expected recipes and the retrieved recipes. The score is between 0
     and infinity. A score of 1 means that the retrieved score is the same as the expected one. A score below 1 means
     that the retrieved score is given factor smaller than the expected one. A score higher 1 means that the retrieved
     score is given factor greater than the expected one. Invalid scores are set to NaN (if there are no expected
     recipes)."""

    metric_name = "fsa_delta_score"

    def compute(
            self,
            query: Optional[str] = None,
            expected_ids: Optional[List[str]] = None,
            retrieved_ids: Optional[List[str]] = None,
            expected_texts: Optional[List[str]] = None,
            retrieved_texts: Optional[List[str]] = None,
            **kwargs: Any,
    ) -> RetrievalMetricResult:
        if "recipes_df" in kwargs and not kwargs["recipes_df"].empty:
            recipes_df = kwargs["recipes_df"]

            expected_recipes = recipes_df[recipes_df["recipe_id"].isin(expected_ids)]
            retrieved_recipes = recipes_df[recipes_df["recipe_id"].isin(retrieved_ids)]

            fsa_score_expected = expected_recipes[["fsa_score"]].mean(axis=0).get(0, np.nan)
            fsa_score_retrieved = retrieved_recipes[["fsa_score"]].mean(axis=0).get(0, np.nan)

            # Difference between the expected and retrieved fsa scores
            if fsa_score_expected == 0:
                score_difference = np.nan
            else:
                score_difference = fsa_score_retrieved / fsa_score_expected

            return RetrievalMetricResult(
                score=score_difference,
                metadata={"metric_name": self.metric_name,
                          "expected_ids": expected_ids,
                          "retrieved_ids": retrieved_ids},
            )
        else:
            raise KeyError("recipes_df not found in kwargs or empty.")


class NUTRIDeltaScoreMetric(BaseRetrievalMetric):
    """Compute the nutri score difference between the expected recipes and the retrieved recipes. The score is between 0
     and infinity. A score of 1 means that the retrieved score is the same as the expected one. A score below 1 means
     that the retrieved score is given factor smaller than the expected one. A score higher 1 means that the retrieved
     score is given factor greater than the expected one. Invalid scores are set to NaN (if there are no expected
     recipes)."""

    metric_name = "nutri_delta_score"

    def compute(
            self,
            query: Optional[str] = None,
            expected_ids: Optional[List[str]] = None,
            retrieved_ids: Optional[List[str]] = None,
            expected_texts: Optional[List[str]] = None,
            retrieved_texts: Optional[List[str]] = None,
            **kwargs: Any,
    ) -> RetrievalMetricResult:
        if "recipes_df" in kwargs and not kwargs["recipes_df"].empty:
            recipes_df = kwargs["recipes_df"]

            expected_recipes = recipes_df[recipes_df["recipe_id"].isin(expected_ids)]
            retrieved_recipes = recipes_df[recipes_df["recipe_id"].isin(retrieved_ids)]

            nutri_score_expected = expected_recipes[["nutri_score"]].mean(axis=0).get(0, np.nan)
            nutri_score_retrieved = retrieved_recipes[["nutri_score"]].mean(axis=0).get(0, np.nan)

            # Difference between the expected and retrieved nutri scores
            if nutri_score_expected == 0:
                score_difference = np.nan
            else:
                score_difference = nutri_score_retrieved / nutri_score_expected

            return RetrievalMetricResult(
                score=score_difference,
                metadata={"metric_name": self.metric_name,
                          "expected_ids": expected_ids,
                          "retrieved_ids": retrieved_ids},
            )
        else:
            raise KeyError("recipes_df not found in kwargs or empty.")
