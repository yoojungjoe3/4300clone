import json
import os
import re
import numpy as np
from helpers.MySQLDatabaseHandler import MySQLDatabaseHandler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# clean query 
def clean_text(query):
    """Convert text to lowercase and remove punctuation."""
    return re.sub(r'[^\w\s]', '', query.lower())

def main(): 
    query = input("Enter your search query: ")  # Asking the user for input

    # Lists to hold extracted data from init.sql
    fandoms = []
    ships = []
    abstract = []
    names = []

    # Regex to capture Name, Fandom, and Ship(s)
    pattern = re.compile(r"VALUES\s*\(\s*'\"(.*?)\"',\s*'\"(.*?)\"',\s*'\"(.*?)\"',\s*'[^']*',\s*'\"[^']*\"',\s*'\"[^']*\"',\s*'\"(.*?)\"'\s*\)")

    # Read the init.sql file and populate the lists
    with open("init.sql", "r", encoding="utf-8") as file:
        for line in file:
            find = pattern.search(line)
            if find:
                names.append(find.group(1))
                fandom_clean = clean_text(find.group(2))
                ship_clean = clean_text(find.group(3))
                abstract_clean = clean_text(find.group(4))
                fandoms.append(fandom_clean)
                ships.append(ship_clean)
                abstract.append(abstract_clean)
                

    def vector_search(query, fandoms, ships, names):
        """
        Compute combined similarity scores for the query against both fandoms and ships.
        Also prints the top two fanfic titles with their similarity values.
        """
    # Clean and prepare the query
    query_text = clean_text(query)
    query_words = query_text.split()
    
    vectorizer = TfidfVectorizer()

    # Create a list to store the similarity scores for the query words
    similarity_scores_fandoms = []
    similarity_scores_ships = []
    similarity_scores_abstracts = []

    # Iterate through each word in the query and compare with fandoms and ships
    for word in query_words:
        #fandom
        join_fandom_query = fandoms + [word]
        tfidf_matrix_fandoms = vectorizer.fit_transform(join_fandom_query)
        query_vector_fandoms = tfidf_matrix_fandoms[-1]
        candidate_vectors_fandoms = tfidf_matrix_fandoms[:-1]
        similarities_fandoms = cosine_similarity(query_vector_fandoms, candidate_vectors_fandoms).flatten()
        similarity_scores_fandoms.append(similarities_fandoms)

        #ship
        join_ship_query = ships + [word]
        tfidf_matrix_ships = vectorizer.fit_transform(join_ship_query)
        query_vector_ships = tfidf_matrix_ships[-1]
        candidate_vectors_ships = tfidf_matrix_ships[:-1]
        similarities_ships = cosine_similarity(query_vector_ships, candidate_vectors_ships).flatten()
        similarity_scores_ships.append(similarities_ships)

        #abstract
        join_abstract_query = abstract + [word]
        tfidf_matrix_abstracts = vectorizer.fit_transform(join_abstract_query)
        query_vector_abstracts = tfidf_matrix_abstracts[-1]
        candidate_vectors_abstracts = tfidf_matrix_abstracts[:-1]
        similarities_abstracts = cosine_similarity(query_vector_abstracts, candidate_vectors_abstracts).flatten()
        similarity_scores_abstracts.append(similarities_abstracts)


    # Combine the similarity scores for each query word (sum of all word similarities)
    combined_fandom_similarities = np.sum(np.array(similarity_scores_fandoms), axis=0)
    combined_ship_similarities = np.sum(np.array(similarity_scores_ships), axis=0)
    combined_abstract_similarities = np.sum(np.array(similarity_scores_abstracts), axis=0)

    # Combine fandom and ship similarities
    combined_similarities = combined_fandom_similarities + combined_ship_similarities + combined_abstract_similarities
    total_sim_dict = {i + 1: total for i, total in enumerate(combined_similarities)}

    # Sort keys (record indices) by similarity score (highest first)
    sorted_keys = sorted(total_sim_dict, key=total_sim_dict.get, reverse=True)

    # Get the keys for the highest and second-highest scores
    highest_key = sorted_keys[0]
    second_highest_key = sorted_keys[1]

    # Adjust index for names list (keys start at 1, list is zero-indexed)
    top_fic = names[sorted_keys[0] - 1] if sorted_keys else None
    second_fic = names[sorted_keys[1] - 1] if len(sorted_keys) > 1 else None

    print(total_sim_dict)

    # Print similarity values and top results
    print(f"Top result: {top_fic} with similarity score: {total_sim_dict[highest_key]}")
    print(f"Second result: {second_fic} with similarity score: {total_sim_dict[second_highest_key]}")

    return total_sim_dict, top_fic, second_fic

if __name__ == "__main__":
    main()
