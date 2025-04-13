import json
import os
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class Entry:
    def __init__(self, name, ship, fandom, rating, abstract, link):
        self.name = name
        self.ship = ship
        self.fandom = fandom
        self.rating = rating
        self.abstract = abstract
        self.link = link

    def to_dict(self):
        return {
            "name": self.name,
            "ship": self.ship,
            "fandom": self.fandom,
            "rating": self.rating,
            "abstract": self.abstract,
            "link": self.link
        }

    def __repr__(self):
        return f"Entry(Name: {self.name}, Ships: {self.ship}, Fandoms: {self.fandom}, Ratings: {self.rating}, Abstracts: {self.abstract}, Links: {self.link})"

def clean_text(query):
    """Convert text to lowercase and remove punctuation."""
    return re.sub(r'[^\w\s]', '', query.lower())

def load_data(filename):
    """
    Reads the init.sql file and populates lists for names, fandoms, ships, reviews, and abstracts.
    Since the regex only extracts five groups, ratings and links are filled with placeholders.
    """
    names = []
    fandoms = []
    ships = []
    ratings = []  # Placeholder values
    links = []    # Placeholder values
    reviews = []
    abstracts = []

    # This pattern expects lines like:
    # VALUES ( '"Name"', '"Fandom"', '"Ship"', '...', '"..."', '"Review"', '"Abstract"' )
    # Adjust the regex if your file uses a different format.
    pattern = re.compile(r"VALUES\s*\(\s*'\"(.*?)\"',\s*'\"(.*?)\"',\s*'\"(.*?)\"',\s*'[^']*',\s*'\"[^']*\"',\s*'\"(.*?)\"',\s*'\"(.*?)\"'\s*\)")
    
    with open(filename, "r", encoding="utf-8") as file:
        for line in file:
            find = pattern.search(line)
            if find:
                names.append(find.group(1))
                fandom_clean = clean_text(find.group(2))
                ship_clean = clean_text(find.group(3))
                review_clean = clean_text(find.group(4))
                abstract_clean = clean_text(find.group(5))
                fandoms.append(fandom_clean)
                ships.append(ship_clean)
                reviews.append(review_clean)
                abstracts.append(abstract_clean)
                # For fields not captured by regex, use a placeholder
                ratings.append("N/A")
                links.append("N/A")
    return names, fandoms, ships, ratings, links, reviews, abstracts

def vector_search(query, names, fandoms, ships, ratings, links, reviews, abstracts):
    """
    Compute combined similarity scores for the query against fandoms, ships, reviews, and abstracts.
    Returns a list of Entry objects sorted by descending similarity.
    """
    # Clean and prepare the query.
    query_text = clean_text(query)
    query_words = query_text.split()
    
    vectorizer = TfidfVectorizer()
    
    # Prepare lists for the similarity scores from each field.
    similarity_scores_fandoms = []
    similarity_scores_ships = []
    similarity_scores_abstracts = []
    similarity_scores_reviews = []
    
    # Iterate through every word in the query and compute similarity against each field.
    for word in query_words:
        # Fandom similarity
        join_fandom_query = fandoms + [word]
        tfidf_matrix_fandoms = vectorizer.fit_transform(join_fandom_query)
        query_vector_fandoms = tfidf_matrix_fandoms[-1]
        candidate_vectors_fandoms = tfidf_matrix_fandoms[:-1]
        sims_fandom = cosine_similarity(query_vector_fandoms, candidate_vectors_fandoms).flatten()
        similarity_scores_fandoms.append(sims_fandom)
        
        # Ship similarity
        join_ship_query = ships + [word]
        tfidf_matrix_ships = vectorizer.fit_transform(join_ship_query)
        query_vector_ships = tfidf_matrix_ships[-1]
        candidate_vectors_ships = tfidf_matrix_ships[:-1]
        sims_ship = cosine_similarity(query_vector_ships, candidate_vectors_ships).flatten()
        similarity_scores_ships.append(sims_ship)
        
        # Abstract similarity
        join_abstract_query = abstracts + [word]
        tfidf_matrix_abstracts = vectorizer.fit_transform(join_abstract_query)
        query_vector_abstracts = tfidf_matrix_abstracts[-1]
        candidate_vectors_abstracts = tfidf_matrix_abstracts[:-1]
        sims_abstract = cosine_similarity(query_vector_abstracts, candidate_vectors_abstracts).flatten()
        similarity_scores_abstracts.append(sims_abstract)
        
        # Review similarity
        join_reviews_query = reviews + [word]
        tfidf_matrix_reviews = vectorizer.fit_transform(join_reviews_query)
        query_vector_reviews = tfidf_matrix_reviews[-1]
        candidate_vectors_reviews = tfidf_matrix_reviews[:-1]
        sims_review = cosine_similarity(query_vector_reviews, candidate_vectors_reviews).flatten()
        similarity_scores_reviews.append(sims_review)
    
    # Sum the similarity scores for each field across all query words.
    combined_fandom_similarities = np.sum(np.array(similarity_scores_fandoms), axis=0)
    combined_ship_similarities = np.sum(np.array(similarity_scores_ships), axis=0)
    combined_abstract_similarities = np.sum(np.array(similarity_scores_abstracts), axis=0)
    combined_review_similarities = np.sum(np.array(similarity_scores_reviews), axis=0)
    
    # Total similarity is the sum from all fields.
    combined_similarities = (combined_fandom_similarities +
                             combined_ship_similarities +
                             combined_abstract_similarities +
                             combined_review_similarities)
    
    # Build a dictionary mapping record index (starting at 1) to similarity score.
    total_sim_dict = {i + 1: sim for i, sim in enumerate(combined_similarities)}
    
    # Sort the record indices by similarity (highest first).
    sorted_keys = sorted(total_sim_dict, key=total_sim_dict.get, reverse=True)
    
    ourentries = []
    for key in sorted_keys:
        # Only consider entries with a nonzero similarity.
        if total_sim_dict[key] != 0:
            # Adjust index by subtracting one since our lists are zero-indexed.
            final_name = names[key-1]
            final_ship = ships[key-1]
            final_fandom = fandoms[key-1]
            final_rating = ratings[key-1]
            final_abstract = abstracts[key-1]
            final_link = links[key-1]
            entry = Entry(final_name, final_ship, final_fandom, final_rating, final_abstract, final_link)
            ourentries.append(entry)
    
    return ourentries

def main():
    query = input("Enter your search query: ")
    
    # Load the data from the init.sql file.
    # Make sure init.sql is in the same directory (or provide an appropriate path).
    names, fandoms, ships, ratings, links, reviews, abstracts = load_data("init.sql")
    
    # Perform the vector search.
    results = vector_search(query, names, fandoms, ships, ratings, links, reviews, abstracts)
    
    print("\nSearch results:")
    for entry in results:
        print(entry)

if __name__ == "__main__":
    main()
