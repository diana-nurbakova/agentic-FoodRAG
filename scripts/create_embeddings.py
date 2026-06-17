# Execute: nohup python3 -u create_embeddings.py > output.log 2>&1 &
# Open output log: cat output.log

print("Importing libraries...")

from llama_index.core import Settings
from llama_index.embeddings.litellm import LiteLLMEmbedding
from llama_index.llms.litellm import LiteLLM

from foodrag.utils import load_util

# Output Path
data_path = './data'
qa_name = 'big_1'

print("Loading configs and LLM/Embedding model...")

# Load configs
nutrition_config = load_util.get_config('config/nutrition_config.yml')
persona_config = load_util.get_config('config/persona_config.yml')

# Resolve dataset locations from the manifest (config/data_manifest.yml) and
# validate everything required is present before the (expensive) init.
from foodrag.data import DataLoader
from foodrag.config import settings as foodrag_settings

_loader = DataLoader.from_manifest()
_loader.require()
data_location = str(_loader.data_location)
graph_location = str(_loader.graph_location)
additional_location = str(_loader.additional_location)

# Init LLM + embedding from config/models.yml + .env (DeepInfra/OpenRouter/HF).
llm = foodrag_settings.build_llm()
embed_model = foodrag_settings.build_embedding()
foodrag_settings.apply_global_settings(llm=llm, embed_model=embed_model)

# Embedding output dir keyed by the active embedding model id (avoids clashes
# when switching models).
_emb_model = foodrag_settings.load_models_config().embedding['model']
embedding_path = data_path + '/embeddings/' + _emb_model.replace('/', '_')

print("Init hummus data...")

# Load (init if necessary) the data; NOTE: create random meal types annotations for the test set
hummus_data = load_util.get_hummus(data_location, graph_location, additional_location, data_path=data_path,
                                   discard_non_tagged_recipes=False, n_clusters=-1, k_user=2, k_recipe=2,
                                   food_class_strategy='None', add_additional_meal_type_labels=True,
                                   clean_food_category=True, similarity_thresholds=False, include_usda=True,
                                   do_annotate_meal_type=True, create_random_meal_types=False, llm=llm)
