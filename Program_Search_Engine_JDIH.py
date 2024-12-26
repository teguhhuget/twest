import re
import pandas as pd
import numpy as np
import string
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from symspellpy import SymSpell
import psycopg2

# Function to preprocess text
def preprocess_text(text):
    if text is None:  # Handle None values
        return ""
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Load typo correction dictionary
sym_spell = SymSpell(max_dictionary_edit_distance=3, prefix_length=7)
dictionary_path = "C:/Users/sutih/Downloads/kamuss.txt"
sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)

# Function to correct typos in user queries
def correct_typos(text):
    # Separate numbers from words
    words = text.split()
    corrected_words = []
    for word in words:
        if re.fullmatch(r'\d+', word):  # Skip numbers
            corrected_words.append(word)
        else:
            # Perform typo correction only for non-numeric words
            suggestions = sym_spell.lookup_compound(word, max_edit_distance=3)
            if suggestions:
                corrected_words.append(suggestions[0].term)
            else:
                corrected_words.append(word)
    corrected_text = " ".join(corrected_words)
    return corrected_text, text

# Load datasets from PostgreSQL
conn = psycopg2.connect(
    host="10.255.240.250",
    port="5432",
    user="postgres",
    password="dika1234",
    database="serdesk_new"
)
cursor = conn.cursor()

# Load solutions_data
cursor.execute("""
    SELECT 
        s.id AS solution_id,
        c.kategori AS kategori,
        sub.subkategori AS subkategori,
        s.judul AS title,
        s.deskripsi AS content,
        s.deleted_at AS deleted
    FROM solutions s
    LEFT JOIN keywords k ON s.keyword_id = k.id
    LEFT JOIN subcategories sub ON k.subcategory_id = sub.id
    LEFT JOIN categories c ON sub.category_id = c.id
    WHERE s.deleted_at IS NULL
""")
solutions_data = [
    {
        "id": row[0],
        "kategori": row[1],
        "subkategori": row[2],
        "title": row[3],
        "content": row[4],
        "deleted": row[5]
    } for row in cursor.fetchall()
]

# Load api_knowledge_jdih_data
cursor.execute("""
    SELECT 
        "idData",
        "judul",
        "sumber",
        "subjek",
        "status"
    FROM api_knowledge_jdih
""")
api_knowledge_jdih_data = [
    {
        "id": row[0],
        "judul": row[1],
        "sumber": row[2],
        "subjek": row[3],
        "status": row[4]
    } for row in cursor.fetchall() if row[4] != "Tidak Berlaku"
]

cursor.close()
conn.close()

print(f"DATA SOLUTION : {len(solutions_data)}")
print(f"DATA JDIH     : {len(api_knowledge_jdih_data)}")

# Combine datasets
data = solutions_data + api_knowledge_jdih_data

# Load SentenceTransformer model
model = SentenceTransformer('indobenchmark/indobert-large-p2')

# Precompute embeddings
solutions_embeddings = [
    model.encode(preprocess_text(article['title'] + " " + article['content']), convert_to_numpy=True)
    for article in solutions_data
]
jdih_embeddings = [
    model.encode(
        preprocess_text(
            (article.get('judul') or '') + " " +
            (article.get('sumber') or '') + " " +
            (article.get('subjek') or '')
        ),
        convert_to_numpy=True
    )
    for article in api_knowledge_jdih_data
]

# Manual search
def search_dataset(keyword, max_results=30):
    keyword_pattern = rf'\b{re.escape(keyword)}\b'
    results = []

    # Search in solutions_data
    for entry in solutions_data:
        if (re.search(keyword_pattern, entry['kategori'], re.IGNORECASE) or
            re.search(keyword_pattern, entry['subkategori'], re.IGNORECASE) or
            re.search(keyword_pattern, entry['title'], re.IGNORECASE) or
            re.search(keyword_pattern, entry['content'], re.IGNORECASE)):
            results.append(entry)
            if len(results) == max_results:
                break

    # Search in api_knowledge_jdih_data
    for entry in api_knowledge_jdih_data:
        sumber = entry.get('sumber', '') or ''
        subjek = entry.get('subjek', '') or ''
        judul = entry.get('judul', '') or ''

        if (re.search(keyword_pattern, judul, re.IGNORECASE) or
            re.search(keyword_pattern, sumber, re.IGNORECASE) or
            re.search(keyword_pattern, subjek, re.IGNORECASE)):
            results.append(entry)
            if len(results) == max_results:
                break

    return results

# Semantic search
def search(query, top_n=10):
    corrected_query, original_query = correct_typos(query)
    if corrected_query != original_query:
        print(f"Kata yang dikoreksi: '{original_query}' menjadi '{corrected_query}'")

    query_embedding = model.encode(preprocess_text(corrected_query), convert_to_numpy=True)
    solutions_similarities = cosine_similarity([query_embedding], solutions_embeddings)[0]
    jdih_similarities = cosine_similarity([query_embedding], jdih_embeddings)[0]

    # Combine results
    solutions_results = [
        {'data': solutions_data[i], 'score': solutions_similarities[i]} for i in range(len(solutions_data))
    ]
    jdih_results = [
        {'data': api_knowledge_jdih_data[i], 'score': jdih_similarities[i]} for i in range(len(api_knowledge_jdih_data))
    ]

    # Sort and filter results
    solutions_results = sorted(solutions_results, key=lambda x: x['score'], reverse=True)
    jdih_results = sorted(jdih_results, key=lambda x: x['score'], reverse=True)
    return solutions_results[:top_n] if solutions_results[0]['score'] >= jdih_results[0]['score'] else jdih_results[:top_n]

# Combine IDs from manual and semantic searches
def get_combined_ids(query):
    manual_results = search_dataset(query)
    semantic_results = search(query)

    combined_ids = []
    seen_ids = set()

    # Add manual result IDs
    for entry in manual_results:
        if entry['id'] not in seen_ids:
            seen_ids.add(entry['id'])
            combined_ids.append(entry['id'])

    # Add semantic result IDs
    for result in semantic_results:
        result_id = result['data']['id']
        if result_id not in seen_ids:
            seen_ids.add(result_id)
            combined_ids.append(result_id)

    return combined_ids

# Main logic
if __name__ == "__main__":
    while True:
        user_query = input("Masukkan kueri pencarian (ketik 'exit' atau 'quit' untuk keluar): ")
        if user_query.lower() in ['exit', 'quit']:
            print("Terima kasih telah menggunakan program pencarian!")
            break

        # Get combined IDs
        combined_ids = get_combined_ids(user_query)

        # Display combined IDs
        if combined_ids:
            print(f"Combined IDs: {combined_ids}")
        else:
            print("Tidak ada hasil ditemukan.")
