#!/usr/bin/python3
import re
import nltk
import sys
import getopt
import heapq
import math
import unicodedata

from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
stemmer = PorterStemmer()

# Settings ###############################################################################

debug = True

# Generic helpers ########################################################################

def dprint(*args, **kwargs):
    if debug:
        print("##", *args, file=sys.stderr, **kwargs)

# Main code ##############################################################################

def usage():
    print("usage: " + sys.argv[0] + " -d dictionary-file -p postings-file -q file-of-queries -o output-file-of-results")

def run_search(dict_file, postings_file, query_file, results_file):
    """
    using the given dictionary file and postings file,
    perform searching on the given queries file and output the results to a file
    """
    print('Running search on the queries...')

    dictionary = load_dictionary(dict_file)
    doc_lengths = load_doc_lengths(postings_file)
    total_docs = len(doc_lengths)

    with (open(query_file, 'r', encoding="utf8") as qfile,
          open(results_file, 'w', encoding="utf8") as rfile):
        query = qfile.readline().strip()
        ranked_results = compute_tfidf_scores(query, dictionary, postings_file, doc_lengths, total_docs)
        rfile.write(' '.join(map(str, ranked_results)) + '\n')
    print("Search completed!")

# Load dictionary
def load_dictionary(dict_file):
    dictionary = {}
    with open(dict_file, 'r', encoding="utf8") as file:
        for line in file:
            term, offset, length = line.strip().split()
            dictionary[term] = (int(offset), int(length))
    return dictionary

# Load document lengths
def load_doc_lengths(postings_file):
    doc_lengths = {}
    with open(postings_file, 'r', encoding="utf8") as file:
        for line in file:
            if line.startswith("LC "):
                _field, docID, length = line.strip().split()
                doc_lengths[(int(docID), 'content')] = float(length)
            else:
                break
    return doc_lengths

# Get posting list
def get_postings_list(file, offset):
    file.seek(offset)
    postings = file.readline().strip().split()
    postings = [(int(p.split(':')[0]), int(p.split(':')[1])) for p in postings]
    return postings

def preprocess_query(query):
    words = []

    # Since we will be doing the bare minimum of treating this as a freetext query for now,
    # remove all special tokens from the query.
    query = query.replace('"', '').replace(' AND ','')

    query = unicodedata.normalize('NFKD', query)

    sentences = sent_tokenize(query)
    for sentence in sentences:
        words.extend([stemmer.stem(word.lower()) for word in word_tokenize(sentence)])
    return words

# Main function for calculating cosine and retrieve the ranked results
def compute_tfidf_scores(query, dictionary, postings_file, doc_lengths, total_docs):
    query_terms = [f"C:{query}" for query in preprocess_query(query)]
    query_tf = {}

    for term in query_terms:
        query_tf[term] = query_tf.get(term, 0) + 1

    scores = {}
    with open(postings_file, 'r', encoding="utf8") as p_file:
        for term in set(query_terms):
            if term in dictionary:
                offset, df = dictionary[term]
                postings = get_postings_list(p_file, offset)

                idf = math.log10(total_docs / df)
                query_weight = (1 + math.log10(query_tf[term])) * idf

                for docID, tf in postings:
                    doc_weight = 1 + math.log10(tf)
                    if docID not in scores:
                        scores[docID] = 0
                    scores[docID] += query_weight * doc_weight  # Compute dot product

    # Normalize scores using document length
    for docID in scores:
        scores[docID] /= doc_lengths[(docID, 'content')]

    # Return results in ranked order
    return sorted(scores.keys(), key=lambda docid: scores[docid], reverse=True)

dictionary_file = postings_file = file_of_queries = output_file_of_results = None

nltk.download('punkt_tab')

try:
    opts, args = getopt.getopt(sys.argv[1:], 'd:p:q:o:v')
except getopt.GetoptError:
    usage()
    sys.exit(2)

for o, a in opts:
    if o == '-d':
        dictionary_file  = a
    elif o == '-p':
        postings_file = a
    elif o == '-q':
        file_of_queries = a
    elif o == '-o':
        file_of_output = a
    elif o == '-v': # verbose mode
        debug = True
    else:
        assert False, "unhandled option"

if dictionary_file == None or postings_file == None or file_of_queries == None or file_of_output == None :
    usage()
    sys.exit(2)

run_search(dictionary_file, postings_file, file_of_queries, file_of_output)
