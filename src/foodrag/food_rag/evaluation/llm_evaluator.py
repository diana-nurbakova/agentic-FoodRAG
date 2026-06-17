import asyncio

import logging
import time
from llama_index.core.bridge.pydantic import BaseModel, Field
from llama_index.core.evaluation import CorrectnessEvaluator, GuidelineEvaluator, RelevancyEvaluator, \
    FaithfulnessEvaluator
from llama_index.core.evaluation.base import EvaluationResult
from typing import Any, Optional, Sequence
from typing import cast

logger = logging.getLogger(__name__)


class EvaluationData(BaseModel):
    passing: bool = Field(description="Whether the response passes the guidelines.")
    feedback: str = Field(
        description="The feedback for the response based on the guidelines."
    )


class SyncCorrectnessEvaluator(CorrectnessEvaluator):
    """Specifically provides synchronous evaluation which does not use asyncio."""

    def evaluate(
            self,
            query: Optional[str] = None,
            response: Optional[str] = None,
            contexts: Optional[Sequence[str]] = None,
            reference: Optional[str] = None,
            sleep_time_in_seconds: int = 0,
            **kwargs: Any,
    ) -> EvaluationResult:
        """Copied aevaluate method from CorrectnessEvaluator and modified it to be synchronous."""
        del kwargs  # Unused
        del contexts  # Unused

        time.sleep(sleep_time_in_seconds)

        if query is None or response is None:
            raise ValueError("query, and response must be provided")

        eval_response = self._llm.predict(
            prompt=self._eval_template,
            query=query,
            generated_answer=response,
            reference_answer=reference or "(NO REFERENCE ANSWER SUPPLIED)",
        )

        # Use the parser function
        score, reasoning = self.parser_function(eval_response)

        return EvaluationResult(
            query=query,
            response=response,
            passing=score >= self._score_threshold if score is not None else None,
            score=score,
            feedback=reasoning,
        )


class SyncGuidelineEvaluator(GuidelineEvaluator):
    """Specifically provides synchronous evaluation which does not use asyncio."""

    def evaluate(
            self,
            query: Optional[str] = None,
            response: Optional[str] = None,
            contexts: Optional[Sequence[str]] = None,
            sleep_time_in_seconds: int = 0,
            **kwargs: Any,
    ) -> EvaluationResult:
        """Copied aevaluate method from GuidelineEvaluator and modified it to be synchronous."""
        del contexts  # Unused
        del kwargs  # Unused
        if query is None or response is None:
            raise ValueError("query and response must be provided")

        logger.debug("prompt: %s", self._eval_template)
        logger.debug("query: %s", query)
        logger.debug("response: %s", response)
        logger.debug("guidelines: %s", self._guidelines)

        time.sleep(sleep_time_in_seconds)

        eval_response = self._llm.predict(
            self._eval_template,
            query=query,
            response=response,
            guidelines=self._guidelines,
        )
        eval_data = self._output_parser.parse(eval_response)
        eval_data = cast(EvaluationData, eval_data)

        return EvaluationResult(
            query=query,
            response=response,
            passing=eval_data.passing,
            score=1.0 if eval_data.passing else 0.0,
            feedback=eval_data.feedback,
        )


class SyncRelevancyEvaluator(RelevancyEvaluator):
    """Specifically provides synchronous evaluation which does not use asyncio. Also, avoids using Document and
     SummaryIndex. Thus, only not too long contexts are supported."""

    async def aevaluate(
            self,
            query: str | None = None,
            response: str | None = None,
            contexts: Sequence[str] | None = None,
            sleep_time_in_seconds: int = 0,
            **kwargs: Any,
    ) -> EvaluationResult:
        """Copied aevaluate method from RelevancyEvaluator and modified it not use SummaryIndex."""
        del kwargs  # Unused

        if query is None or contexts is None or response is None:
            raise ValueError("query, contexts, and response must be provided")

        query_response = f"Question: {query}\nResponse: {response}"

        await asyncio.sleep(sleep_time_in_seconds)

        # Main eval
        eval_response_1 = await self._llm.apredict(
            prompt=self._eval_template,
            query_str=query_response,
            context_str='\n'.join(contexts),
        )

        await asyncio.sleep(sleep_time_in_seconds)

        # Refine eval
        eval_response_2 = await self._llm.apredict(
            prompt=self._refine_template,
            query_str=query_response,
            existing_answer=eval_response_1,
            context_msg='\n'.join(contexts),
        )

        if "yes" in eval_response_2.lower():
            passing = True
        else:
            if self._raise_error:
                raise ValueError("The response is invalid")
            passing = False

        return EvaluationResult(
            query=query,
            response=response,
            passing=passing,
            score=1.0 if passing else 0.0,
            feedback=eval_response_2,
            contexts=contexts,
        )

    def evaluate(
            self,
            query: str | None = None,
            response: str | None = None,
            contexts: Sequence[str] | None = None,
            sleep_time_in_seconds: int = 0,
            **kwargs: Any,
    ) -> EvaluationResult:
        """Copied aevaluate method from RelevancyEvaluator and modified it to be synchronous."""
        del kwargs  # Unused

        if query is None or contexts is None or response is None:
            raise ValueError("query, contexts, and response must be provided")

        query_response = f"Question: {query}\nResponse: {response}"

        time.sleep(sleep_time_in_seconds)

        # Main eval
        eval_response_1 = self._llm.predict(
            prompt=self._eval_template,
            query_str=query_response,
            context_str='\n'.join(contexts),
        )

        time.sleep(sleep_time_in_seconds)

        # Refine eval
        eval_response_2 = self._llm.predict(
            prompt=self._refine_template,
            query_str=query_response,
            existing_answer=eval_response_1,
            context_msg='\n'.join(contexts),
        )

        if "yes" in eval_response_2.lower():
            passing = True
        else:
            if self._raise_error:
                raise ValueError("The response is invalid")
            passing = False

        return EvaluationResult(
            query=query,
            response=response,
            passing=passing,
            score=1.0 if passing else 0.0,
            feedback=eval_response_2,
            contexts=contexts,
        )


class SyncFaithfulnessEvaluator(FaithfulnessEvaluator):
    """Specifically provides synchronous evaluation which does not use asyncio. Also, avoids using Document and
     SummaryIndex. Thus, only not too long contexts are supported."""

    async def aevaluate(
            self,
            query: str | None = None,
            response: str | None = None,
            contexts: Sequence[str] | None = None,
            sleep_time_in_seconds: int = 0,
            **kwargs: Any,
    ) -> EvaluationResult:
        """Copied aevaluate method from FaithfulnessEvaluator and modified it to not use SummaryIndex."""
        del kwargs  # Unused

        await asyncio.sleep(sleep_time_in_seconds)

        if contexts is None or response is None:
            raise ValueError("contexts and response must be provided")

        # Main eval
        eval_response_1 = await self._llm.apredict(
            prompt=self._eval_template,
            query_str=response,
            context_str='\n'.join(contexts),
        )

        await asyncio.sleep(sleep_time_in_seconds)

        # Refine eval
        eval_response_2 = await self._llm.apredict(
            prompt=self._refine_template,
            query_str=response,
            existing_answer=eval_response_1,
            context_msg='\n'.join(contexts),
        )

        if "yes" in eval_response_2.lower():
            passing = True
        else:
            passing = False
            if self._raise_error:
                raise ValueError("The response is invalid")

        return EvaluationResult(
            query=query,
            response=response,
            contexts=contexts,
            passing=passing,
            score=1.0 if passing else 0.0,
            feedback=eval_response_2,
        )


    def evaluate(
            self,
            query: str | None = None,
            response: str | None = None,
            contexts: Sequence[str] | None = None,
            sleep_time_in_seconds: int = 0,
            **kwargs: Any,
    ) -> EvaluationResult:
        """Copied aevaluate method from FaithfulnessEvaluator and modified it to be synchronous."""
        del kwargs  # Unused

        time.sleep(sleep_time_in_seconds)

        if contexts is None or response is None:
            raise ValueError("contexts and response must be provided")

        # Main eval
        eval_response_1 = self._llm.predict(
            prompt=self._eval_template,
            query_str=response,
            context_str='\n'.join(contexts),
        )

        time.sleep(sleep_time_in_seconds)

        # Refine eval
        eval_response_2 = self._llm.predict(
            prompt=self._refine_template,
            query_str=response,
            existing_answer=eval_response_1,
            context_msg='\n'.join(contexts),
        )

        if "yes" in eval_response_2.lower():
            passing = True
        else:
            passing = False
            if self._raise_error:
                raise ValueError("The response is invalid")

        return EvaluationResult(
            query=query,
            response=response,
            contexts=contexts,
            passing=passing,
            score=1.0 if passing else 0.0,
            feedback=eval_response_2,
        )
