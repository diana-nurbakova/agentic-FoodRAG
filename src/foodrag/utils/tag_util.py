from itertools import chain

import pandas as pd

DEFAULT_TAG_URL = 'https://www.food.com/tag/'


def get_tag_from_url(url):
    split_url = url.split(DEFAULT_TAG_URL)

    if len(split_url) > 1:
        return split_url[-1]  # Return the last entry of the url
    else:
        split_url = url.split('/')

        if len(split_url) > 1:
            return split_url[-1]  # Return the last part of a string including back-slashes
        else:  # No splits were successful: Return original string
            return url


def get_url_from_tag(tag):
    return DEFAULT_TAG_URL + str(tag)


def get_tags_string_AND(tag_list):
    return get_tags_string(tag_list, logic_operator='and')


def get_tags_string_OR(tag_list):
    return get_tags_string(tag_list, logic_operator='or')


def get_tags_string(tag_list, logic_operator):
    if isinstance(tag_list, list) and len(tag_list) > 1:
        comma_separated_part = ', '.join([tag for tag in tag_list[:-1]])
        tags_string = (' ' + logic_operator + ' ').join([comma_separated_part, tag_list[-1]])
        return tags_string
    else:  # we ignore the case of a single tag and return for single and no tags None
        return None


def remove_non_values(set):
    return {elem for elem in set if elem is not None and not pd.isna(elem)}


def get_tag_recipe_dict(recipe_df):
    return recipe_df[['tags', 'recipe_url']].explode('tags').groupby('tags')['recipe_url'].agg(list).to_dict()


def get_recipe_tag_dict(recipe_df):
    return recipe_df[['recipe_url', 'tags', 'title']].set_index('recipe_url').to_dict(orient='index')


def get_tag_andStringTag_dict(recipe_df, logic_operator):
    if logic_operator == 'and':
        tag_column = 'tag_string_AND'
    elif logic_operator == 'or':
        tag_column = 'tag_string_OR'
    else:
        raise ValueError('logic_operator must be either "and" or "or"')

    exploded_df = recipe_df.explode('tags').groupby('tags')[tag_column].agg(set)
    exploded_df = exploded_df.apply(remove_non_values)  # remove None values in the sets
    exploded_df = exploded_df[exploded_df.apply(lambda x: len(x) > 0)]  # remove rows of empty sets
    tag2tag_strings = exploded_df.to_dict()
    return tag2tag_strings


def get_andStringTag_recipes_dict(recipe_df, logic_operator):
    if logic_operator == 'and':
        tag_column = 'tag_string_AND'
    elif logic_operator == 'or':
        tag_column = 'tag_string_OR'
    else:
        raise ValueError('logic_operator must be either "and" or "or"')

    exploded_df = recipe_df.explode(tag_column).groupby(tag_column)
    title_dict = exploded_df['title'].agg(list).to_dict()
    label_dict = exploded_df['tags'].agg(lambda x: set(chain.from_iterable(x))).to_dict()

    # merge both dicts
    tag_string2recipes = {key: {'dish_names': set(title_dict[key]), 'tag_list': label_dict[key]} for key in title_dict}

    return tag_string2recipes


def get_tag_dicts(recipe_df):
    # get dicts between tags and recipes
    recipe_tag_dict = get_recipe_tag_dict(recipe_df)
    tag_recipe_dict = get_tag_recipe_dict(recipe_df)

    # dict consisting of each tag string as key and a nested dict including respective dish names and tags
    andStringTag_recipes_dict = get_andStringTag_recipes_dict(recipe_df, 'and')

    # dict consisting of each existing tag as key, and all tag combinations present in the recipe df as values
    tag_andStringTag_dict = get_tag_andStringTag_dict(recipe_df, 'and')

    return recipe_tag_dict, tag_recipe_dict, andStringTag_recipes_dict, tag_andStringTag_dict
