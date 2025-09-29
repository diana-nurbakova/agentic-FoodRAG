import psycopg2
from llama_index.vector_stores.postgres import PGVectorStore


class PostgresHelper:
    def __init__(self, credentials, reset=False):
        self.host = credentials['host']
        self.password = credentials['password']
        self.port = credentials['port']
        self.user = credentials['user']
        self.db_name = credentials['db_name']

        if reset:
            self.init()
        else:
            # check if the db with db_name already exists
            try:
                psycopg2.connect(
                    dbname=self.db_name,
                    host=self.host,
                    password=self.password,
                    port=self.port,
                    user=self.user,
                )
            # when the db does not exist or the credentials are wrong, init the db
            except psycopg2.OperationalError:
                self.init()

    # Connects to the standard postgres db and creates a new db with the name db_name (overriding if it already exists)
    def init(self):
        conn = psycopg2.connect(
            dbname="postgres", # standard db in order to remove and create the target db
            host=self.host,
            password=self.password,
            port=self.port,
            user=self.user,
        )
        conn.autocommit = True

        # Init the named db
        with conn.cursor() as c:
            c.execute(f"DROP DATABASE IF EXISTS {self.db_name}")
            c.execute(f"CREATE DATABASE {self.db_name}")
            c.execute("CREATE EXTENSION IF NOT EXISTS vector;")

        return self

    def connect(self, table_name, embed_dim=384):
        # Connect to our named db (Vector Store)
        vector_store = PGVectorStore.from_params(
            database=self.db_name,
            host=self.host,
            password=self.password,
            port=self.port,
            user=self.user,
            table_name=table_name,
            embed_dim=embed_dim,
            hybrid_search=True,
            text_search_config="english",
            hnsw_kwargs={
                "hnsw_m": 16,
                "hnsw_ef_construction": 64,
                "hnsw_ef_search": 40,
                "hnsw_dist_method": "vector_cosine_ops",
            },
        )

        return vector_store
