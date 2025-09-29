import pandas as pd
from abc import ABC, abstractmethod
from llama_index.core.schema import QueryType


class DataFrameRetrieverBase(ABC):
    """
    Abstract base class for retrievers that return DataFrames.
    """

    @abstractmethod
    def retrieve(self, str_or_query_bundle: QueryType) -> pd.DataFrame:
        """
        Retrieves a DataFrame of results for the given query.

        :param QueryType str_or_query_bundle: Query string or QueryBundle (which optionally contains the embedding).
        :returns: Retrieved results as a DataFrame.
        :rtype: pd.DataFrame
        """
        pass

    @abstractmethod
    async def aretrieve(self, str_or_query_bundle: QueryType) -> pd.DataFrame:
        """
        Asynchronously retrieves a DataFrame of results for the given query.

        :param QueryType str_or_query_bundle: Query string or QueryBundle (which optionally contains the embedding).
        :returns: Retrieved results as a DataFrame.
        :rtype: pd.DataFrame
        """
        pass
