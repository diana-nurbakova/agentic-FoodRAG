import numpy as np
from neo4j import GraphDatabase
from pathlib import Path

from foodrag.hummus_transformation import transform_kg
from foodrag.utils import load_util

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

    def insert_data(self, data_path, mode, batch_size=1000, verbose=True, use_absolute_paths=False):
        """
        Loads CSVs into Neo4j and inserts nodes and relationships in a single query. For Cypher load-csv documentation,
        see: https://neo4j.com/docs/cypher-manual/current/clauses/load-csv/

        :param data_path: Path to the data parent directory containing the CSV files.
        :param mode: Mode of the transformation.
        :param batch_size: Batch size for PERIODIC COMMIT (default: 1000).
        :param verbose: If True, prints progress information.
        :param use_absolute_paths: If True, LOAD CSV uses absolute host paths (local Neo4j with
            filesystem access). If False (default), paths are relative to the neo4j_kg import root,
            which must be mounted into the Neo4j container's import dir (see docker-compose.yml).
        """
        # Lazy import here to avoid circular imports
        from foodrag.hummus_transformation.transform_kg import CSV_DELIMITER
        from foodrag.hummus_transformation.transform_kg import SUB_DIRECTORY_NAME, CONFIG_FILE_NAME

        # Get urls
        file_names_dict = load_util.load_file(Path(data_path, SUB_DIRECTORY_NAME, mode, CONFIG_FILE_NAME))

        # By default emit paths relative to the neo4j_kg import root, which is
        # how a Dockerized Neo4j reads them (the host data/neo4j_kg is mounted to
        # the container import dir). Pass use_absolute_paths=True for a local
        # Neo4j that can read the host filesystem directly (the original behavior).
        import_root = Path(data_path, SUB_DIRECTORY_NAME).resolve()

        node_queries = []
        relation_queries = []
        # (label, property) pairs the relation MATCHes look up -- we index these
        # before inserting so the MATCHes use index seeks, not full label scans.
        index_specs = set()
        for name, object_type in file_names_dict['files'].items():
            metadata = load_util.load_file(transform_kg.get_file_metadata_path(data_path, mode, object_type, name))
            file_abs = transform_kg.get_file_path(data_path, mode, object_type, name).resolve()
            csv_ref = file_abs.as_posix() if use_absolute_paths else file_abs.relative_to(import_root).as_posix()

            if object_type == "node":
                query = f"""
                LOAD CSV WITH HEADERS FROM 'file:///{csv_ref}' AS row FIELDTERMINATOR '{CSV_DELIMITER}'
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

                # The MATCHes below look up (src_label, clean(src_id_name)) and
                # (dst_label, clean(dst_id_name)); index those node properties.
                index_specs.add((src_label, Neo4jManager.clean_column_name(src_id_name)))
                index_specs.add((dst_label, Neo4jManager.clean_column_name(dst_id_name)))

                query = f"""
                LOAD CSV WITH HEADERS FROM 'file:///{csv_ref}' AS row FIELDTERMINATOR '{CSV_DELIMITER}'
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
            # 1. Create indexes on the node id-properties the relation MATCHes use.
            # Without these, each MATCH over a 270k-node label is a full scan, so
            # the relation phase (millions of rows) is effectively unbounded.
            for label, prop in sorted(index_specs):
                index_query = f"CREATE INDEX IF NOT EXISTS FOR (n:`{label}`) ON (n.`{prop}`)"
                if verbose:
                    print(f"Executing:{index_query}")
                session.run(index_query)

            # 2. Insert nodes (MERGE can now use the indexes too).
            for query in node_queries:
                if verbose:
                    print(f"Executing:{query}")
                session.run(query)

            # 3. Wait for the indexes to finish populating before the relation
            # MATCHes rely on them.
            session.run("CALL db.awaitIndexes(600)")

            # 4. Insert relationships.
            for query in relation_queries:
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

    def delete_all(self, batch_size=10000):
        """
        Deletes all nodes and relationships in the database.

        Batched via CALL { } IN TRANSACTIONS (a single DETACH DELETE of millions
        of nodes exhausts the transaction) and the result is consumed so the
        delete actually completes before any subsequent insert.
        """
        with self.driver.session() as session:
            session.run(
                f"MATCH (n) CALL (n) {{ DETACH DELETE n }} IN TRANSACTIONS OF {batch_size} ROWS"
            ).consume()

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
