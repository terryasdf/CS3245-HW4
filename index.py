#!/usr/bin/python3
import re
import nltk
import sys
import getopt
import os
import math
import csv
import unicodedata
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from collections import defaultdict
import time
import sys as csv_sys

stemmer = PorterStemmer()

def usage():
    print("usage: " + sys.argv[0] + " -i dataset-file -d dictionary-file -p postings-file")

# Dictionary to store term frequencies - separate dicts for each zone/field
content_index = defaultdict(lambda: defaultdict(int))
title_index = defaultdict(lambda: defaultdict(int))
court_index = defaultdict(lambda: defaultdict(int))
date_index = defaultdict(lambda: defaultdict(int))

# Store document lengths for each zone
content_doc_lengths = {}
title_doc_lengths = {}

# Clean text to handle problematic Unicode characters
def clean_text(text):
    if not text:
        return text
    
    # Normalize Unicode (NFD splits ligatures into component characters)
    text = unicodedata.normalize('NFKD', text)    
    return text

# Process text for a specific field
def process_text(text, doc_id, field_index):
    if not text:  # Handle empty text
        return
    
    # Clean text before processing
    text = clean_text(text)
        
    sentences = sent_tokenize(text)
    for sentence in sentences:
        words = word_tokenize(sentence)
        for word in words:
            stemmed_word = stemmer.stem(word.lower())
            field_index[stemmed_word][doc_id] += 1

# Read and process CSV dataset
def process_dataset(dataset_file):
    print(f"Processing dataset: {dataset_file}")
    
    # Increase CSV field size limit to handle large content fields
    csv_sys.maxsize = 2147483647  # Set to maximum allowed integer
    csv.field_size_limit(min(csv_sys.maxsize, 2147483647))
    
    doc_ids = []
    
    with open(dataset_file, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            doc_id = int(row['document_id'])
            doc_ids.append(doc_id)
            
            # Process each field separately for zone indexing
            process_text(row['content'], doc_id, content_index)
            process_text(row['title'], doc_id, title_index)
            process_text(row['court'], doc_id, court_index)
            
            # Store date as is (for range queries)
            if 'date_posted' in row and row['date_posted']:
                date = row['date_posted'].split()[0]  # Extract just the date part
                date_index[date][doc_id] = 1
    
    # Compute document lengths for content (used for cosine similarity in VSM)
    for doc_id in doc_ids:
        content_doc_lengths[doc_id] = math.sqrt(sum(
            (1 + math.log10(tf))**2 
            for word in content_index 
            for tf in [content_index[word].get(doc_id, 0)] 
            if tf > 0
        ))
        
        # Also compute document lengths for title
        title_doc_lengths[doc_id] = math.sqrt(sum(
            (1 + math.log10(tf))**2 
            for word in title_index 
            for tf in [title_index[word].get(doc_id, 0)] 
            if tf > 0
        ))
    
    return doc_ids

# Write the inverted index to files
def write_index(dict_file, postings_file):
    """ Writes dictionary, postings files, and document lengths with zone information """
    postings_meta = {}
    
    # Specify UTF-8 encoding for output files
    with open(dict_file, 'w', encoding='utf-8') as d_file, open(postings_file, 'w', encoding='utf-8') as p_file:
        # Store doc lengths
        for doc_id, length in content_doc_lengths.items():
            p_file.write(f"LC {doc_id} {length}\n")
        
        for doc_id, length in title_doc_lengths.items():
            p_file.write(f"LT {doc_id} {length}\n")
        
        # Write content index with zone marker "C:"
        sorted_content_terms = sorted(content_index.keys())
        for term in sorted_content_terms:
            postings_meta[f"C:{term}"] = p_file.tell()
            postings = [(doc_id, content_index[term][doc_id]) for doc_id in sorted(content_index[term].keys())]
            p_file.write(' '.join(f"{doc_id}:{tf}" for doc_id, tf in postings) + "\n")
            d_file.write(f"C:{term} {postings_meta[f'C:{term}']} {len(postings)}\n")
        
        # Write title index with zone marker "T:"
        sorted_title_terms = sorted(title_index.keys())
        for term in sorted_title_terms:
            postings_meta[f"T:{term}"] = p_file.tell()
            postings = [(doc_id, title_index[term][doc_id]) for doc_id in sorted(title_index[term].keys())]
            p_file.write(' '.join(f"{doc_id}:{tf}" for doc_id, tf in postings) + "\n")
            d_file.write(f"T:{term} {postings_meta[f'T:{term}']} {len(postings)}\n")
        
        # Write court index with zone marker "COURT:"
        sorted_court_terms = sorted(court_index.keys())
        for term in sorted_court_terms:
            postings_meta[f"COURT:{term}"] = p_file.tell()
            postings = [(doc_id, 1) for doc_id in sorted(court_index[term].keys())]
            p_file.write(' '.join(f"{doc_id}:1" for doc_id, _ in postings) + "\n")
            d_file.write(f"COURT:{term} {postings_meta[f'COURT:{term}']} {len(postings)}\n")
        
        # Write date index with zone marker "DATE:"
        sorted_dates = sorted(date_index.keys())
        for date in sorted_dates:
            postings_meta[f"DATE:{date}"] = p_file.tell()
            postings = [(doc_id, 1) for doc_id in sorted(date_index[date].keys())]
            p_file.write(' '.join(f"{doc_id}:1" for doc_id, _ in postings) + "\n")
            d_file.write(f"DATE:{date} {postings_meta[f'DATE:{date}']} {len(postings)}\n")

def build_index(dataset_file, out_dict, out_postings):
    """
    Build index from the dataset file,
    then output the dictionary file and postings file
    """
    print('indexing...')
    start_time = time.time()
    
    doc_ids = process_dataset(dataset_file)
    write_index(out_dict, out_postings)
    
    print("Total documents indexed:", len(doc_ids))
    print("Total unique terms (content):", len(content_index))
    print("Total unique terms (title):", len(title_index))
    print("Total unique courts:", len(court_index))
    print("Total unique dates:", len(date_index))
    print("Done")
    
    end_time = time.time()
    print(f"Indexing completed in {end_time - start_time:.2f} seconds")

dataset_file = output_file_dictionary = output_file_postings = None
debug = False

try:
    opts, args = getopt.getopt(sys.argv[1:], 'i:d:p:v')
except getopt.GetoptError:
    usage()
    sys.exit(2)

for o, a in opts:
    if o == '-i': # dataset file
        dataset_file = a
    elif o == '-d': # dictionary file
        output_file_dictionary = a
    elif o == '-p': # postings file
        output_file_postings = a
    elif o == '-v': # verbose mode
        debug = True
    else:
        assert False, "unhandled option"

if dataset_file == None or output_file_postings == None or output_file_dictionary == None:
    usage()
    sys.exit(2)

build_index(dataset_file, output_file_dictionary, output_file_postings)