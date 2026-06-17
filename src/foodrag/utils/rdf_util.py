from rdflib import URIRef
from tqdm.notebook import tqdm

FOOD_ON_PREFIX = 'http://purl.obolibrary.org/obo/FOODON_'  # Prefix for valid FoodOn products

# Entity Types
FOOD_ON_FOOD_PRODUCT = FOOD_ON_PREFIX + '00001002'
FOOD_ON_CHEESE = FOOD_ON_PREFIX + '00001013'
FOOD_ON_BEVERAGE = FOOD_ON_PREFIX + '03301977'
FOOD_ON_PLANT = FOOD_ON_PREFIX + '00001015'  # But also includes e.g. mushrooms
FOOD_ON_VEGGIE = FOOD_ON_PREFIX + '00001261'
FOOD_ON_FRUIT = FOOD_ON_PREFIX + '03315615'
FOOD_ON_WATER = FOOD_ON_PREFIX + '00002340'

# RDF(S) Properties
RDF_TYPE = 'http://www.w3.org/1999/02/22-rdf-syntax-ns#type'
RDFS_SUBCLASS = 'http://www.w3.org/2000/01/rdf-schema#subClassOf'
RDFS_LABEL = 'http://www.w3.org/2000/01/rdf-schema#label'
RDFS_PROPERTY = 'http://www.w3.org/2000/01/rdf-schema#Property'
RDFS_SUB_PROPERTY = 'http://www.w3.org/2000/01/rdf-schema#subPropertyOf'
RDFS_RANGE = 'http://www.w3.org/2000/01/rdf-schema#range',
RDFS_COMMENT = 'http://www.w3.org/2000/01/rdf-schema#comment'

# OWL Properties
OWL_CLASS = 'http://www.w3.org/2002/07/owl#Class'
OWL_EQUIVALENT_CLASS = 'http://www.w3.org/2002/07/owl#equivalentClass'
OWL_OBJECT_PROPERTY = 'http://www.w3.org/2002/07/owl#ObjectProperty'
OWL_SYMMETRIC_PROPERTY = 'http://www.w3.org/2002/07/owl#SymmetricProperty'
OWL_DISJOINT_WITH = 'http://www.w3.org/2002/07/owl#disjointWith'


class PreProcessedFoodOn:
    def __init__(self, hummus_data):
        # Lazy import to avoid circular imports
        from foodrag.utils import hummus_util

        self.rdf_graph = hummus_data['food_on_ontology']
        self.ingredient_leaves_dict = hummus_util.invert_dict(hummus_data['food_on_mappings'])
        self.ingredient_dict = hummus_util.invert_dict(hummus_data['ingredients_dict'])
        self.ingredient_name_dict = hummus_data['ingredient_names_dict']
        self.all_nodes = None
        self.all_connected_nodes = None
        self.node_name_url_dict = None

    def preprocess(self):
        _ = self.get_all_nodes(remove_unconnected=True, as_url=True)
        _ = self._get_node_name_url_dict()
        return self

    def get_all_nodes(self, remove_unconnected=False, as_url=False):
        """
        Get all nodes in the FoodOn ontology.
        """
        if self.all_nodes:
            nodes = self.all_nodes
        else:
            nodes = self._get_child_nodes(FOOD_ON_FOOD_PRODUCT, depth=None)
            self.all_nodes = nodes

        if remove_unconnected:
            if self.all_connected_nodes:
                nodes = self.all_connected_nodes
            else:
                nodes = [node for node in tqdm(nodes, desc="Preprocessing connected nodes for the first time.") if
                         self.get_ingredient_leaves(node)]
                self.all_connected_nodes = nodes

        if as_url:
            return nodes
        else:
            return self.get_node_names(nodes)

    def get_direct_child_nodes(self, node, as_url=False):
        """
        Get all direct child nodes (leaves and inner nodes) for a given node. Excludes the node itself.
        """
        return self.get_child_nodes(node, as_url=as_url, depth=1)

    def get_child_nodes(self, node, as_url=False, depth=None):
        """
        Get all child nodes (leaves and inner nodes) for a given node. Excludes the node itself.

        :param node: Node for which to get all child nodes
        :param as_url: If True, return URLs instead of node names
        :param depth: Maximum depth to search for child nodes. If None, search until the end.
        """
        nodes = self._get_child_nodes(node, depth)

        if as_url:
            return nodes
        else:
            return self.get_node_names(nodes)

    def _get_child_nodes(self, node, depth=None):
        node = _get_URI(node)
        children = set()

        # Get all direct children
        direct_children = {
            s for s, p, o in self.rdf_graph.triples((None, URIRef(RDFS_SUBCLASS), node))
            if str(s).startswith(FOOD_ON_PREFIX)
        }
        children.update(direct_children)

        # If depth is 1, return the direct children
        if depth is not None and depth <= 1:
            return children

        # Recursively find children of the current children
        new_depth = None if depth is None else depth - 1
        for child in direct_children:
            children.update(self._get_child_nodes(child, new_depth))

        return children

    def get_direct_parent_nodes(self, node, as_url=False):
        """
        Get all direct parent node-paths of a given node. Includes the node itself in each path. Note: If depth is
         smaller than a path to the top node, also includes paths which would not have reached the top node.
        """
        return self.get_parent_nodes(node, as_url=as_url, depth=1)

    def get_parent_nodes(self, node, top_node=FOOD_ON_FOOD_PRODUCT, as_url=False, depth=None):
        """
        Get all parent node-paths until the top node. Includes the node itself in each path. Note: If depth is smaller
        than a path to the top node, also includes paths which would not have reached the top node.
        :param node: Node for which to get all parent nodes
        :param top_node: Top node until which to get all parent nodes
        :param as_url: If True, return URLs instead of node names
        :param depth: Maximum depth to search for parent nodes. If None, search until the top node.
        """
        nodes = self._get_parent_nodes(node, top_node, depth)

        if as_url:
            return nodes
        else:
            return [self.get_node_names(node_path) for node_path in nodes]

    def _get_parent_nodes(self, node, top_node=FOOD_ON_FOOD_PRODUCT, depth=None):
        node = _get_URI(node)
        top_node = _get_URI(top_node)

        # Base case: if the node is the top node, return a path containing only this node
        if node == top_node:
            return [[node]]

        # Get all parent nodes of the current node
        parents = [
            parent for parent in self.rdf_graph.objects(subject=node, predicate=URIRef(RDFS_SUBCLASS))
            if str(parent).startswith(FOOD_ON_PREFIX)
        ]

        # If there are no parents, return an empty list (no path to the top node)
        if not parents:
            return []

        # If depth is 1, return the direct parents paths
        if depth is not None and depth <= 1:
            return [[node, parent] for parent in parents]

        # Recursively find all paths for each parent node
        all_paths = []
        new_depth = None if depth is None else depth - 1
        for parent in parents:
            parent_paths = self._get_parent_nodes(parent, top_node, new_depth)

            for path in parent_paths:
                all_paths.append([node] + path)

        return all_paths

    def get_ingredient_leaves(self, node):
        """
        Get all valid connected food leaves for a given node. A leave is the food.com ingredient connected to all children.
        """
        # Get all children of the node
        node = _get_URI(node)
        all_children = self._get_child_nodes(node, depth=None)

        # Get all ingredient leaves of the children
        ingredient_urls = set(
            ingredient
            for sublist in [self.ingredient_leaves_dict.get(str(child), []) for child in all_children]
            for ingredient in sublist
        )
        ingredient_ids = [self.ingredient_dict[url] for url in ingredient_urls if url in self.ingredient_dict]
        ingredient_names = [self.ingredient_name_dict[ingredient_id] for ingredient_id in ingredient_ids]

        return ingredient_names

    def get_node_names(self, nodes):
        """
        Get the names of one or more nodes using their RDFS_LABEL.
        If a node has no label, return its string representation.

        :param nodes: List of nodes (URIRef)
        :return: A single name (if one node is passed) or a list of names (if multiple nodes are passed)
        """
        # Handle a single node or multiple nodes and convert nodes to uris if necessary
        if not isinstance(nodes, list):
            nodes = list(nodes)
        nodes = [_get_URI(node) for node in nodes]

        node_names = []
        for node in nodes:
            label = self.rdf_graph.value(subject=node, predicate=URIRef(RDFS_LABEL), any=True)
            node_names.append(str(label) if label else str(node))

        return node_names

    def translate_ingredients(self, ingredients, include_food_on, include_food_com):
        """
        Translate ingredients from FoodOn and/or Food.com names to Food.com ingredient names.

        :param ingredients: List of ingredient names
        :param include_food_on: If FoodOn food products are included in the ingredients
        :param include_food_com: If Food.com ingredient names are included in the ingredients
        :return: List Food.com ingredient names
        """
        translated_ingredients = []
        node_name_url_dict = self._get_node_name_url_dict()

        for ingredient in ingredients:
            # Translate FoodOn food products to Food.com ingredient names
            if include_food_on:
                ingredient_url = node_name_url_dict.get(ingredient, None)

                if ingredient_url:
                    food_on_ingredient_names = self.get_ingredient_leaves(ingredient_url)
                    translated_ingredients.extend(food_on_ingredient_names)

            if include_food_com:
                translated_ingredients.append(ingredient)

        return translated_ingredients

    def _get_node_name_url_dict(self):
        """
        Get a dictionary mapping node names to their URLs.
        """
        if self.node_name_url_dict:
            return self.node_name_url_dict
        else:
            nodes = self.get_all_nodes(as_url=True)
            node_names = self.get_node_names(nodes)
            self.node_name_url_dict = dict(zip(node_names, nodes))

            return self.node_name_url_dict


def _get_URI(node):
    if isinstance(node, str):
        node = URIRef(node)
    elif isinstance(node, URIRef):
        pass
    else:
        raise ValueError(f"Node must be either of type URIRef or str, but was {type(node)}.")
    return node
