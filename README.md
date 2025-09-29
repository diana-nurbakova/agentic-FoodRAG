# Food RAG

Repository of the food RAG system based on the HUMMUS KG [[1](#1)].
The system is based/uses LlamaIndex [[2](#2)] and Nebula Graph [[3](#3)].

## Installation & Usage:

1. Clone the repository
2. Install python 3.11.9 (preferably with conda)
3. Install the requirements with `pip install -r requirements.txt`
4. Preprocess the data
    1. Download the [HummusKG dataset](https://gitlab.com/felix134/connected-recipe-data-set)
    2. Extract the dataset and provide its path in all to-be-used Jupyter Notebooks
    3. Create a question-answer dataset for retriever evaluation later on
       1. Execute `qa_generation_1.ipynb` to create a question-answer dataset for evaluation (has to be done only once). For
          the second question-answer dataset, including only ingredient class requirements (instead of direct inferred
          ingredients), execute `qa_generation_2.ipynb`.
       2. The question-answer dataset will be saved in the `data/qa` folder
    4. Execute `create_embeddings.ipynb` to preprocess the embeddings (has to be done only once)
    5. Initialize the stores:
        1. Execute `init_vector_store.ipynb` to preprocess the db (has to be done only once)
        2. Execute `init_graph_store_1.ipynb` and optionally `init_graph_store_2.ipynb` to preprocess the KGs (has to be done only
           once, and only for the desired graphs)
    6. The preprocessed Hummus KG will be saved in the `data/hummus` folder, the Nebula KG in the `data/nebula_kg`
       folder, the graph store in the `data/graph_store` folder, and the embeddings in the  `data/embeddings` folder
5. Run the Food RAG system by following the notebook's instructions: `food_RAG_agents.ipynb`.
6. For the complete evaluation refer to the `eval_1.ipynb` and the `eval_2.ipynb` notebooks. The evaluation results will be saved in the `data/runs`
   folder. Alternatively, you can run the evaluation script with `python eval_1.py`.

## System Overview

![img.png](img/food_rag_overview.png)

## Evaluation Overview

![img.png](img/food_rag_eval.png)

### Notes

- Vanilla RAG: query augmentation not used.
- What kind of graph (for non vanilla RAG) might be helpful for our Meal Plan Recommender
- Focus on Vector Store first and how to retrieve results

# Resources

[<a id="1">1</a>] Felix Bölz, Diana Nurbakova, Sylvie Calabretto, Armin Gerl, Lionel Brunie, and Harald Kosch. 2023.
HUMMUS: A Linked, Healthiness-Aware, User-centered and Argument-Enabling Recipe Data Set for Recommendation. In
Proceedings of the 17th ACM Conference on Recommender Systems (RecSys '23). Association for Computing Machinery, New
York, NY, USA, 1–11. https://doi.org/10.1145/3604915.3609491

[<a id="2">2</a>] LlamaIndex Repository. https://docs.llamaindex.ai/en/stable/

[<a id="3">3</a>]  Nebula Graph Repository. https://www.nebula-graph.io/