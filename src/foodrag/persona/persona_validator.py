def validate_metadata(metadata_collection, persona) -> bool:
    """Validate if the metadata is valid regarding the persona."""
    if persona is not None:
        # Check tags
        if _check_tags(metadata_collection, persona):
            # Check ingredients
            if _check_ingredients(metadata_collection, persona):
                # Check relative_nutrients
                if _check_relative_nutrients(metadata_collection, persona):
                    # Check total_nutrients
                    if _check_total_nutrients(metadata_collection, persona):
                        # Check nutrition_scores
                        return _check_nutrition_scores(metadata_collection, persona)

        # If any of the checks failed, return False
        return False
    else:
        return True


def _check_tags(metadata_collection, persona):
    if persona['tags'] is not None:
        for metadata in metadata_collection:
            tags = set(metadata.get('tags', []))
            tag_likes = set(persona['tags'].get('likes', []))
            tag_dislikes = set(persona['tags'].get('dislikes', []))

            # Here both likes and dislikes are hard constraints, meaning they always must be present/absent
            valid_tags = tags.intersection(tag_likes) == tag_likes and tag_dislikes.difference(tags) == tag_dislikes

            if not valid_tags:
                return False

    return True


def _check_ingredients(metadata_collection, persona):
    if persona['ingredients'] is not None:
        for metadata in metadata_collection:
            ingredients = set(metadata.get('ingredient_names', []))
            ingredient_likes = set(persona['ingredients'].get('likes', []))
            ingredient_dislikes = set(persona['ingredients'].get('dislikes', []))

            # Here both likes and dislikes are hard constraints, meaning they always must be present/absent
            valid_ingredients = ingredients.intersection(
                ingredient_likes) == ingredient_likes and ingredient_dislikes.difference(
                ingredients) == ingredient_dislikes

            if not valid_ingredients:
                return False

    return True


def _check_relative_nutrients(metadata_collection, persona):
    if persona['relative_nutrients'] is not None:
        for metadata in metadata_collection:
            for nutrient, values in persona['relative_nutrients'].items():
                if nutrient in metadata:
                    nutrient_value = metadata[nutrient]
                    limits = values['limits']

                    # Check if the actual nutrient is within the limits
                    valid_relative_nutrients = limits['lower'] <= nutrient_value <= limits['upper']

                    if not valid_relative_nutrients:
                        return False

    return True


def _check_total_nutrients(metadata_collection, persona): # TODO does it correctly check the total nutrients?
    if persona['total_nutrients'] is not None:
        nutrient_sum = {}

        for metadata in metadata_collection:
            for nutrient, value in persona['total_nutrients'].items():
                if nutrient in metadata:
                    nutrient_value = metadata[nutrient]

                    # Add the nutrient to the total nutrients
                    if nutrient in nutrient_sum:
                        nutrient_sum[nutrient] += nutrient_value
                    else:
                        nutrient_sum[nutrient] = nutrient_value

        # Check if the sum of each nutrient is within its limits
        for nutrient, total_value in nutrient_sum.items():
            valid_total_nutrients = total_value <= persona['total_nutrients'][nutrient]

            if not valid_total_nutrients:
                return False

    return True


def _check_nutrition_scores(metadata_collection, persona):
    if persona['nutrition_scores'] is not None:
        for metadata in metadata_collection:
            for score, value in persona['nutrition_scores'].items():
                if score in metadata:
                    score_value = metadata[score]

                    # Check if the actual score is bigger (==better/healthier) than the limit
                    valid_nutrition_scores = score_value >= value

                    if not valid_nutrition_scores:
                        return False

    return True


def get_valid_nodes(nodes, persona):
    valid_nodes = [node for node in nodes if validate_metadata([node.metadata], persona)]

    return valid_nodes

