# Execute: nohup python3 -u eval_1.py > output.log 2>&1 &
# Open output log: cat output.log

print("Importing libraries...")

from pathlib import Path
from llama_index.core import Settings
from llama_index.core import StorageContext
from llama_index.core import VectorStoreIndex
from llama_index.core.indices.knowledge_graph import KGTableRetriever
from llama_index.core.prompts.base import PromptTemplate, PromptType
from llama_index.embeddings.litellm import LiteLLMEmbedding
from llama_index.graph_stores.nebula import NebulaGraphStore
from llama_index.llms.litellm import LiteLLM

from foodrag.food_rag.utils import nebula_helper, qa_helper, embedding_helper, hummus_docify, postgres_helper
from foodrag.food_rag.evaluation import retriever_evaluator
from foodrag.food_rag.retriever import advanced_retriever
from foodrag.food_rag.retriever import db_retriever
from foodrag.food_rag.retriever import kg_retriever
from foodrag.food_rag.retriever import pandas_retriever
from foodrag.food_rag.retriever import wrapper_retriever
from foodrag.persona import qa_generator
from foodrag.utils import load_util
from foodrag.utils import network_util
from foodrag.food_rag.utils.embedding_helper import QA_SUB_PATH
from foodrag.food_rag.evaluation.retriever_evaluator import RUN_PATH
from foodrag.utils import hummus_util

## Output Paths
data_path = './data'
embedding_path = data_path + '/embeddings/snowflake_l'
qa_name = 'small_1'
run_name = 'eval_1'

print("Loading configs and LLM/Emnedding model...")

# Load configs
nutrition_config = load_util.get_config('config/nutrition_config.yml')
persona_config = load_util.get_config('config/persona_config.yml')

# Load key
api_key = load_util.load_file('config/api_key')

# Init LLM
llm = LiteLLM(  # Doc see @ --> https://docs.litellm.ai/docs/providers/openai_compatible
    api_base="https://llms-inference.innkube.fim.uni-passau.de",
    api_key=api_key,
    model="openai/llama3.1",
    max_retries=100,
    cooldown_time=60,
)

embed_model = LiteLLMEmbedding(  # Doc see @ --> https://docs.litellm.ai/docs/providers/openai_compatible
    api_base="https://llms-inference.innkube.fim.uni-passau.de",
    api_key=api_key,
    model_name="openai/Snowflake/snowflake-arctic-embed-l",
)

# Set Global Settings
Settings.llm = llm
Settings.embed_model = embed_model
Settings.chunk_size = 512

print("Init KG and postgres...")

# Postgres credentials
postgres_credentials = {
    'db_name': 'rag_postgres',
    'host': 'localhost',
    'password': 'postgres',
    'port': '5432',
    'user': 'postgres',
}

# Connect to the default 'postgres' database
vector_store = postgres_helper.PostgresHelper(postgres_credentials, reset=False).connect(table_name='hummus_recipes',
                                                                                         embed_dim=1024)
vector_store_index = VectorStoreIndex.from_vector_store(vector_store)

# Nebula schema
# Load the preprocessed KG data
nebula_kg = transform_kg.Neo4JTransformer(path=data_path)
nebula_schema = nebula_kg.get_schema(config_path='old_code/old_config/nebula/string_indices/nebula_import_small.yml')

# Get Nebula KG store parameters
ip_address, port, user, password = network_util.set_nebula_env_vars()

# Connect to the Nebula KG store
graph_store = NebulaGraphStore(
    space_name=nebula_schema['space_name'],
    edge_types=nebula_schema['edge_types'],
    rel_prop_names=nebula_schema['edge_features'],
    tags=nebula_schema['vertex_types'],
    tag_prop_names=nebula_schema['vertex_features']
)
storage_context = StorageContext.from_defaults(graph_store=graph_store)

# Init the graph store with the alternative space
graph_store_2 = nebula_helper.load_index(data_path + '/graph_store', "Alternative_RecipesSpace", ["relationship"],
                                         ["relationship"],
                                         ["entity"], verbose=False)
storage_context_2 = StorageContext.from_defaults(graph_store=graph_store_2)

print("Loading data...")

# Load the current recipes data
hummus_data = load_util.get_current_hummus(data_path)
recipes_df = hummus_data['recipes_df']

# Load question-answer pairs & their embeddings (only the recipe qa's)
question_answer_sets = qa_generator.load_qa_set(data_path, sub_dir_name=qa_name, recipes_df=recipes_df)
qa_recipe_doc_dict, _ = embedding_helper.load_existing_batches(embedding_path + QA_SUB_PATH + qa_name + '/qa_recipe_documents')

# Clean the qa set and add the embeddings
qa_set = qa_helper.get_best_questions(question_answer_sets['recipe_question_answer_set'], discard_empty_qas=True)
qa_set = qa_helper.extend_qa_embeddings(qa_set, qa_recipe_doc_dict)

# Set the default metrics
metrics = ["hit_rate", "mrr", "ndcg", "precision", "recall", "f1", "who_score", "fsa_score", "nutri_score",
           "who_delta_score", "fsa_delta_score", "nutri_delta_score"]

# Get dataset general nutrition scores
valid_nutrition_scores = qa_helper.get_qa_nutrition_scores(qa_set, recipes_df)

print("Evaluating retriever_1...")
# Set up the retriever # TODO similarity_top_k=100 if set 10 the results are not good --> makes sense because the qa is not of fixed length and not ordered. How to solve?
retriever_1 = db_retriever.RecipeRetriever(index=vector_store_index, recipes_df=recipes_df,
                                           vector_store_query_mode="default", similarity_top_k=100, split_node_ratio=2)

# Set up the evaluator
evaluator_1 = retriever_evaluator.DataFrameRetrieverEvaluator.from_metric_names(
    metric_names=metrics, retriever=retriever_1, recipes_df=recipes_df
)

# Evaluate
response_1, metrics_1 = evaluator_1.evaluate_dataset(qa_set=qa_set, data_path=data_path, sub_dir_name=run_name + '/retriever_1', batch_size=10, show_progress=True)
print(metrics_1)

print("Evaluating retriever_2...")
# Set up the retriever
retriever_2 = db_retriever.RecipeRetriever(index=vector_store_index, recipes_df=recipes_df,
                                           vector_store_query_mode="sparse", similarity_top_k=100, split_node_ratio=2)

# Set up the evaluator
evaluator_2 = retriever_evaluator.DataFrameRetrieverEvaluator.from_metric_names(
    metric_names=metrics, retriever=retriever_2, recipes_df=recipes_df
)

# Evaluate
response_2, metrics_2 = evaluator_2.evaluate_dataset(qa_set=qa_set, data_path=data_path, sub_dir_name=run_name + '/retriever_2', batch_size=10, show_progress=True)
print(metrics_2)

print("Evaluating retriever_3...")
# Set up the retriever
retriever_3 = db_retriever.RecipeRetriever(index=vector_store_index, recipes_df=recipes_df,
                                           vector_store_query_mode="hybrid", similarity_top_k=100, split_node_ratio=2)

# Set up the evaluator
evaluator_3 = retriever_evaluator.DataFrameRetrieverEvaluator.from_metric_names(
    metric_names=metrics, retriever=retriever_3, recipes_df=recipes_df
)

# Evaluate
response_3, metrics_3 = evaluator_3.evaluate_dataset(qa_set=qa_set, data_path=data_path, sub_dir_name=run_name + '/retriever_3', batch_size=10, show_progress=True)
print(metrics_3)

print("Evaluating retriever_4...")
# Set up the retriever
retriever_4 = db_retriever.RecipeRetriever(index=vector_store_index, recipes_df=recipes_df,
                                           vector_store_query_mode="text_search", similarity_top_k=100,
                                           split_node_ratio=2)

# Set up the evaluator
evaluator_4 = retriever_evaluator.DataFrameRetrieverEvaluator.from_metric_names(
    metric_names=metrics, retriever=retriever_4, recipes_df=recipes_df
)

# Evaluate
response_4, metrics_4 = evaluator_4.evaluate_dataset(qa_set=qa_set, data_path=data_path, sub_dir_name=run_name + '/retriever_4', batch_size=10, show_progress=True)
print(metrics_4)

print("Evaluating retriever_5...")
# Clean df for the pandas retriever
cleaned_recipes_df = recipes_df.copy()
cleaned_recipes_df = hummus_docify.stringify_nested_lists(cleaned_recipes_df)
cleaned_recipes_df = hummus_docify.stringify_nested_dicts(cleaned_recipes_df)
cleaned_recipes_df = cleaned_recipes_df.rename(
    columns={'duration': 'duration [min]', 'ingredients': 'ingredient_quantities',
             'ingredient_names': 'ingredients', 'direction_size': 'number_of_directions',
             'ingredients_sizes': 'number_of_ingredients'})
cleaned_recipes_df = cleaned_recipes_df.drop(
    columns=['last_changed_date', 'food_kg_locator', 'recipe_url', 'normalization_comment'])
cleaned_recipes_df = hummus_util.clean_serves_to_int(cleaned_recipes_df)

# Set up the retriever
retriever_5 = pandas_retriever.PandasQueryRetriever(df=cleaned_recipes_df, verbose=False)

# Set up the evaluator
evaluator_5 = retriever_evaluator.DataFrameRetrieverEvaluator.from_metric_names(
    metric_names=metrics, retriever=retriever_5, recipes_df=cleaned_recipes_df
)

# Evaluate
response_5, metrics_5 = evaluator_5.evaluate_dataset(qa_set=qa_set, data_path=data_path, sub_dir_name=run_name + '/retriever_5', batch_size=10, show_progress=True)
print(metrics_5)

print("Evaluating retriever_6...")
# TODO Experiment with the synthesis prompt a bit
graph_query_synthesis_prompt = PromptTemplate("""Task: Generate a NebulaGraph query from natural language input.

Instructions:
- Vertex IDs are strings representing names or titles. So instead of v.Vertex.name == 'Foo', use id(v) == 'Foo'.
- Use only the provided relationship types, tags, and properties from the schema.
- Do not alter any relationship types, tags, or properties, including capitalization.
- Return only the Cypher query string (that can be directly executed in NebulaGraph).
- Do not add explanations, notes, apologies, or anything but the query in your responses.
- Do not enclose the result by ```cypher or ```.
- Always return the list of recipe_urls as answer.
- Keep the query firstly performant, secondly simple, and thirdly reduce the usage of MATCH and WHERE clauses.

Schema:
- Space Name: {space_name},
- Vertices: {tags},
- Edges: {edges},
- Vertex Property Names: {tag_properties},
- Edge Property Names: {edge_properties},

Notes:
- Scores (who score, fsa score, nutri score) are normalized between 0 and 1, with 1 being the highest.
- Default score is the nutri score.
- Recipes are unhealthy with a score < 0.3, neutral with a score ≥ 0.3, and healthy with a score ≥ 0.5.
- If there are nutrition requirements (e.g. amount of sugar), always use the properties given for the recipes themselves.
- If nutrition requirements are given as a range, use two conditions. For example, if sugar should be between 10 and 100, use r.Recipe.sugars > 10 AND r.Recipe.sugars < 100.
- NebulaGraph Cypher (nGQL) uses:
    1. == for comparison.
    2. Explicit vertex specification for vertex properties. For example, instead of v.property, use v.Foo.property if Foo is the vertex.

    Example for both, the following is valid NebulaGraph Cypher dialect:
        MATCH (r:Recipe)-[:has_ingredient]->(i:Ingredient) WHERE id(i) == 'Ginger'
        RETURN r.Recipe.recipe_url;

Question: {query_str}

Answer (Cypher Query):""", prompt_type=PromptType.TEXT_TO_GRAPH_QUERY)

# Build the query engine
retriever_6 = kg_retriever.KGQueryEngine(
    recipes_df=recipes_df,
    storage_context=storage_context,
    llm=llm,
    verbose=False,
    graph_query_synthesis_prompt=graph_query_synthesis_prompt,
    **nebula_schema
)

# Set up the evaluator
evaluator_6 = retriever_evaluator.DataFrameRetrieverEvaluator.from_metric_names(
    metric_names=metrics, retriever=retriever_6, recipes_df=recipes_df
)

# Evaluate
response_6, metrics_6 = evaluator_6.evaluate_dataset(qa_set=qa_set, data_path=data_path, sub_dir_name=run_name + '/retriever_6', batch_size=10, show_progress=True)
print(metrics_6)

print("Evaluating retriever_7...")
# Set up the retriever
graph_store_2_retriever: KGTableRetriever = graph_store_2.as_retriever()
graph_store_2_retriever.num_chunks_per_query = 100
retriever_7 = wrapper_retriever.WrapperNodeRecipeRetriever(base_retriever=graph_store_2_retriever,
                                                           recipes_df=recipes_df)

# Set up the evaluator
evaluator_7 = retriever_evaluator.DataFrameRetrieverEvaluator.from_metric_names(
    metric_names=metrics, retriever=retriever_7, recipes_df=recipes_df
)

# Evaluate
response_7, metrics_7 = evaluator_7.evaluate_dataset(qa_set=qa_set, data_path=data_path, sub_dir_name=run_name + '/retriever_7', batch_size=10, show_progress=True)
print(metrics_7)

print("Evaluating retriever_8...")
# Set up the retriever
fused_retrievers = [retriever_6, retriever_7]
retriever_8 = advanced_retriever.DataFrameFusionRetriever(retrievers=fused_retrievers, recipes_df=recipes_df,
                                                          mode="simple", num_queries=3, use_async=False,
                                                          similarity_top_k=100, verbose=True)

# Set up the evaluator
evaluator_8 = retriever_evaluator.DataFrameRetrieverEvaluator.from_metric_names(
    metric_names=metrics, retriever=retriever_8, recipes_df=recipes_df
)

# Evaluate
response_8, metrics_8 = evaluator_8.evaluate_dataset(qa_set=qa_set, data_path=data_path, sub_dir_name=run_name + '/retriever_8', batch_size=10, show_progress=True)
print(metrics_8)

print("Evaluating retriever_9...")
# Set up the retriever
retriever_9 = advanced_retriever.IntersectionRetriever(retriever_5, retriever_7, verbose=True)

# Set up the evaluator
evaluator_9 = retriever_evaluator.DataFrameRetrieverEvaluator.from_metric_names(
    metric_names=metrics, retriever=retriever_9, recipes_df=recipes_df
)

# Evaluate
response_9, metrics_9 = evaluator_9.evaluate_dataset(qa_set=qa_set, data_path=data_path, sub_dir_name=run_name + '/retriever_9', batch_size=10, show_progress=True)
print(metrics_9)

print("Evaluating retriever_10...")
# Set up the retriever
maj_int_retrievers = [retriever_5, retriever_7, retriever_2]
vote_weights = {0: 0.3, 1: 0.3, 2: 0.3}  # Results must match at least 2 retrievers
retriever_10 = advanced_retriever.MajorityIntersectionRetriever(retrievers=maj_int_retrievers,
                                                                vote_weights=vote_weights, verbose=True)

# Set up the evaluator
evaluator_10 = retriever_evaluator.DataFrameRetrieverEvaluator.from_metric_names(
    metric_names=metrics, retriever=retriever_10, recipes_df=recipes_df
)

# Evaluate
response_10, metrics_10 = evaluator_10.evaluate_dataset(qa_set=qa_set, data_path=data_path, sub_dir_name=run_name + '/retriever_10', batch_size=10, show_progress=True)
print(metrics_10)

print("Storing data...")
# Summarize the evaluation results in a df
results_df = retriever_evaluator.summarize_results(
    {
        'Dense Vector Store Retriever (1)': metrics_1,
        'Sparse Vector Store Retriever (2)': metrics_2,
        'Hybrid Vector Store Retriever (3)': metrics_3,
        'Text Search Vector Store Retriever (4)': metrics_4,
        'Text-to-Pandas Retriever (5)': metrics_5,
        'Text-to-Cipher (Nebula) Retriever (6)': metrics_6,
        'Dense Nebula Retriever (7)': metrics_7,
        'Fusion Retriever (8)': metrics_8,
        'Intersection Retriever (9)': metrics_9,
        'Majority Intersection Retriever (10)': metrics_10
    }
)

# Store the evaluation results
load_util.store_data(data={
    'evaluation_results': results_df,
    'response_1': [response['retrieved_ids'] for response in response_1],
    'response_2': [response['retrieved_ids'] for response in response_2],
    'response_3': [response['retrieved_ids'] for response in response_3],
    'response_4': [response['retrieved_ids'] for response in response_4],
    'response_5': [response['retrieved_ids'] for response in response_5],
    'response_6': [response['retrieved_ids'] for response in response_6],
    'response_7': [response['retrieved_ids'] for response in response_7],
    'response_8': [response['retrieved_ids'] for response in response_8],
    'response_9': [response['retrieved_ids'] for response in response_9],
    'response_10': [response['retrieved_ids'] for response in response_10],
}, data_path=Path(data_path, RUN_PATH, run_name))
