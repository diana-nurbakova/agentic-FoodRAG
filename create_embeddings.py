# Execute: nohup python3 -u create_embeddings.py > output.log 2>&1 &
# Open output log: cat output.log

print("Importing libraries...")

from llama_index.core import Settings
from llama_index.embeddings.litellm import LiteLLMEmbedding
from llama_index.llms.litellm import LiteLLM

from src.utils import load_util

# Output Path
data_path = './data'
embedding_path = data_path + '/embeddings/jina_v2_base_de'
qa_name = 'big_1'

## Raw HUMMUS data
data_location = './data/init/'
graph_location = './data/init/graph/'
additional_location = './data/init/additional/'

print("Loading configs and LLM/Emedding model...")

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

# Remote Embedding model
embed_model = LiteLLMEmbedding(  # Doc see @ --> https://docs.litellm.ai/docs/providers/openai_compatible
    api_base="https://llms-inference.innkube.fim.uni-passau.de",
    api_key=api_key,
    model_name="jinaai/jina-embeddings-v2-base-de",
)

# Set Global Settings
Settings.llm = llm
Settings.embed_model = embed_model
Settings.chunk_size = 512

print("Init hummus data...")

# Load (init if necessary) the data; NOTE: create random meal types annotations for the test set
hummus_data = load_util.get_hummus(data_location, graph_location, additional_location, data_path=data_path,
                                   discard_non_tagged_recipes=False, n_clusters=-1, k_user=2, k_recipe=2,
                                   food_class_strategy='None', add_additional_meal_type_labels=True,
                                   clean_food_category=True, similarity_thresholds=False, include_usda=True,
                                   do_annotate_meal_type=True, create_random_meal_types=False, llm=llm)
