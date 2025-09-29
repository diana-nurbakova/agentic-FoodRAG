import numpy as np
from neo4j import GraphDatabase
from pathlib import Path

from src.hummus_transformation import transform_kg
from src.utils import load_util

# Refer to: https://neo4j.com/docs/cypher-manual/current/values-and-types/casting-data/
NEO4J_CASTING_FUNCTIONS = {
    np.floating: 'toFloat',
    np.integer: 'toInteger',
    np.bool_: 'toBoolean',
}


class Neo4jManager:
    def __init__(self, uri, user, password, database):
        """
        Initializes Neo4j connection.

        :param uri: Neo4j Bolt URI
        :param user: Neo4j username
        :param password: Neo4j password
        :param database: Neo4j database name
        """
        self.driver = GraphDatabase.driver(uri, auth=(user, password), database=database)

    def close(self):
        """Closes the Neo4j connection."""
        self.driver.close()

    def insert_data(self, data_path, mode, batch_size=1000, verbose=True):
        """
        Loads CSVs into Neo4j and inserts nodes and relationships in a single query. For Cypher load-csv documentation,
        see: https://neo4j.com/docs/cypher-manual/current/clauses/load-csv/

        :param data_path: Path to the data parent directory containing the CSV files.
        :param mode: Mode of the transformation.
        :param batch_size: Batch size for PERIODIC COMMIT (default: 1000).
        :param verbose: If True, prints progress information.
        """
        # Lazy import here to avoid circular imports
        from src.hummus_transformation.transform_kg import CSV_DELIMITER
        from src.hummus_transformation.transform_kg import SUB_DIRECTORY_NAME, CONFIG_FILE_NAME

        # Get urls
        file_names_dict = load_util.load_file(Path(data_path, SUB_DIRECTORY_NAME, mode, CONFIG_FILE_NAME))

        node_queries = []
        relation_queries = []
        for name, object_type in file_names_dict['files'].items():
            metadata = load_util.load_file(transform_kg.get_file_metadata_path(data_path, mode, object_type, name))
            absolute_file_path = transform_kg.get_file_path(data_path, mode, object_type, name).resolve().as_posix()

            if object_type == "node":
                query = f"""
                LOAD CSV WITH HEADERS FROM 'file:///{absolute_file_path}' AS row FIELDTERMINATOR '{CSV_DELIMITER}'
                CALL (row) {{
                    MERGE (n:{metadata['node_type']} {self._to_import_attr(metadata=metadata,
                                                                           left_include='{',
                                                                           right_include='}', )})
                }} IN TRANSACTIONS OF {batch_size} ROWS;
                """
                node_queries.append(query)
            elif object_type == "relation":
                relation_type = metadata["relation_type"]
                src_label, dst_label = metadata["src_node_type"], metadata["dst_node_type"]
                src_id_col, dst_id_col = metadata["src_id_column"], metadata["dst_id_column"]
                src_id_name, dst_id_name = metadata["src_id_name"], metadata["dst_id_name"]

                query = f"""
                LOAD CSV WITH HEADERS FROM 'file:///{absolute_file_path}' AS row FIELDTERMINATOR '{CSV_DELIMITER}'
                CALL (row) {{
                    MATCH (a:{src_label} {{ {Neo4jManager._get_casting_function(np.integer, src_id_col, node_id_name=src_id_name)} }})
                    MATCH (b:{dst_label} {{ {Neo4jManager._get_casting_function(np.integer, dst_id_col, node_id_name=dst_id_name)} }})
                    MERGE (a)-[r:{relation_type} {self._to_import_attr(metadata=metadata,
                                                                       left_include='{',
                                                                       right_include='}',
                                                                       excluded_headers=[src_id_col, dst_id_col])}]->(b)
                }} IN TRANSACTIONS OF {batch_size} ROWS;
                """
                relation_queries.append(query)
            else:
                if verbose:
                    print(f"Skipping {name} file as type {type} is not recognized: {str(object_type)}")

        with self.driver.session() as session:
            for query in node_queries + relation_queries:
                if verbose:
                    print(f"Executing:{query}")
                session.run(query)

    @staticmethod
    def _to_import_attr(metadata, excluded_headers=None, left_include='', right_include='',
                        use_includes_when_empty=False):
        if excluded_headers is None:
            excluded_headers = []

        headers = [header for header in metadata['headers'] if header not in excluded_headers]
        dtypes = metadata['dtypes']
        casted_headers = [Neo4jManager._get_casting_function(dtype=dtypes.get(header, None), id_name=header) for header
                          in headers]

        if not casted_headers:  # if empty
            return f"{left_include} {', '.join(casted_headers)} {right_include}" if use_includes_when_empty else ""
        return f"{left_include} {', '.join(casted_headers)} {right_include}"

    @staticmethod
    def _get_casting_function(dtype, id_name, node_id_name=None):
        if node_id_name is None:
            node_id_name = id_name

        cleaned_name = Neo4jManager.clean_column_name(node_id_name)
        feature_cast_tmpl = "{cleaned_name}: {cast_func}(row['{name}'])"
        feature_tmpl = "{cleaned_name}: row['{name}']"
        cast_func = None

        if dtype is not None:
            for np_type, type_cast_func in NEO4J_CASTING_FUNCTIONS.items():
                if hasattr(dtype, 'type'):  # if the type is from pandas
                    dtype = dtype.type
                if np.issubdtype(dtype, np_type):
                    cast_func = type_cast_func

        if cast_func is not None:
            return feature_cast_tmpl.format(cleaned_name=cleaned_name, name=id_name, cast_func=cast_func)
        else:
            return feature_tmpl.format(cleaned_name=cleaned_name, name=id_name)

    def delete_all(self):
        """
        Deletes all nodes and relationships in the database.
        """
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")

    def query(self, query):
        """
        Runs a Cypher query and returns the result.

        :param query: Cypher query to run
        :return: Result of the query
        """
        with self.driver.session() as session:
            result = session.run(query)
            return result.data()

    @staticmethod
    def clean_column_name(old_name):
        # get the whole name until the first space
        return old_name.split(' ')[0]
