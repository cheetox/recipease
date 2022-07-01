#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from xd.ingredient_parser import ingredient_parser
import pickle
import unidecode, ast


# Top-N recomendations order by score

# In[3]:


def get_recommendations(N, scores):
    # load in recipe dataset
    df_recipes = pd.read_csv("df_parsed.csv")
    # order the scores with and filter to get the highest N scores
    top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:N]
    # create dataframe to load in recommendations
    recommendation = pd.DataFrame(columns=["recipe", "ingredients", "score", "url"])
    count = 0
    for i in top:
        recommendation.at[count, "recipe"] = title_parser(df_recipes["recipe_name"][i])
        recommendation.at[count, "ingredients"] = ingredient_parser_final(
            df_recipes["ingredients"][i]
        )
        recommendation.at[count, "url"] = df_recipes["recipe_urls"][i]
        recommendation.at[count, "score"] = "{:.3f}".format(float(scores[i]))
        count += 1
    return recommendation


# neaten the ingredients being outputted

# In[4]:


def ingredient_parser_final(ingredient):
    if isinstance(ingredient, list):
        ingredients = ingredient
    else:
        ingredients = ast.literal_eval(ingredient)
    ingredients = ",".join(ingredients)
    ingredients = unidecode.unidecode(ingredients)
    return ingredients


# In[5]:


def title_parser(title):
    title = unidecode.unidecode(title)
    return title


# In[6]:


def RecSys(ingredients, N=5):
    """
    The reccomendation system takes in a list of ingredients and returns a list of top 5 
    recipes based of of cosine similarity. 
    ingredients: a list of ingredients
    N: the number of reccomendations returned 
    return: top 5 reccomendations for cooking recipes
    """

    # load in tdidf model and encodings
    with open("tfidf_encodings.pkl", "rb") as f:
        tfidf_encodings = pickle.load(f)
    with open("tfidf.pkl", "rb") as f:
        tfidf = pickle.load(f)

    # parse the ingredients using my ingredient_parser
    try:
        ingredients_parsed = ingredient_parser(ingredients)
    except:
        ingredients_parsed = ingredient_parser([ingredients])

    # use our pretrained tfidf model to encode our input ingredients
    ingredients_parsed = " ".join(ingredients_parsed)
    ingredients_tfidf = tfidf.transform([ingredients_parsed])

    # calculate cosine similarity between actual recipe ingreds and test ingreds
    cos_sim = map(lambda x: cosine_similarity(ingredients_tfidf, x), tfidf_encodings)
    scores = list(cos_sim)

    # Filter top N recommendations
    recommendations = get_recommendations(N, scores)
    return recommendations


# In[7]:


RecSys(["tomato","ground beef","onion"],5)


# In[9]:


RecSys(["tomato","ground beef","onion"],5).iloc [0, 1]


# In[ ]:




