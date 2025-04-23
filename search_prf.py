#!/usr/bin/python3
import re
import nltk
import sys
import getopt
import math
import unicodedata
from collections import defaultdict, Counter

from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
stemmer = PorterStemmer()

# Settings (Number of)

TOP_K_DOCS = 5      #  top docs to assume relevant

EXPAND_TERMS = 10    # expansion terms to add

TITLE_WT = 5.0       # Title weighting factor

def usage():
    print("usage: {} -d dictionary-file -p postings-file -q file-of-query -o output-file".format(sys.argv[0]))

def load_dictionary(dict_file):
    dictionary = {}
    with open(dict_file, 'r', encoding="utf8") as file:
        for line in file:
            term, offset, df = line.strip().split()
            dictionary[term] = (int(offset), int(df))
    return dictionary

def load_doc_lengths(postings_file):
    doc_lengths = {}
    with open(postings_file, 'r', encoding="utf8") as file:
        for line in file:
            parts = line.strip().split()
            if parts[0] in ('LC', 'LT'):
                field = 'content' if parts[0]=='LC' else 'title'
                docID, length = int(parts[1]), float(parts[2])
                doc_lengths[(docID, field)] = length
            else:
                break
    return doc_lengths

def get_postings_list(p_file, offset):
    p_file.seek(offset)
    postings = p_file.readline().strip().split()
    return [(int(p.split(':')[0]), int(p.split(':')[1])) for p in postings]

def preprocess(query):
    query = query.replace('"','').replace(' AND ', ' ')
    query = unicodedata.normalize('NFKD', query)
    tokens = []
    for sent in sent_tokenize(query):
        tokens.extend([stemmer.stem(t.lower()) for t in word_tokenize(sent)])
    return tokens

def compute_scores(query_terms, dictionary, postings_file, doc_lengths, total_docs):
    tf_q = Counter(query_terms)
    # term weights in query
    wq = {t: (1+math.log10(tf_q[t])) * math.log10(total_docs/dictionary[f"C:{t}"][1])
          for t in tf_q if f"C:{t}" in dictionary}

    content_scores = defaultdict(float)
    title_scores = defaultdict(float)
    with open(postings_file,'r',encoding="utf8") as pf:
        for term, qw in wq.items():
            for zone in ('C', 'T'):
                key = f"{zone}:{term}"
                if key not in dictionary: continue
                offset, df = dictionary[key]
                postings = get_postings_list(pf, offset)
                idf = math.log10(total_docs/df)
                for docID, tf in postings:
                    wt = (1+math.log10(tf))*idf
                    if zone=='C': content_scores[docID] += wq[term]*wt
                    else: title_scores[docID] += wq[term]*wt
    scores = {}
    for d, sc in content_scores.items():
        scores[d] = sc / doc_lengths[(d,'content')]
    for d, st in title_scores.items():
        scores[d] = scores.get(d,0) + (st/ doc_lengths[(d,'title')])*TITLE_WT
    return scores

def expand_query(orig_terms, top_docs, dictionary, postings_file, doc_lengths, total_docs):
    term_scores = defaultdict(float)
    # for each doc in top k, accumulate term*idf
    with open(postings_file,'r',encoding="utf8") as pf:
        for key,(offset,df) in dictionary.items():
            if not key.startswith('C:'): continue
            term = key.split(':',1)[1]
            postings = get_postings_list(pf, offset)
            idf = math.log10(total_docs/df)
            for docID, tf in postings:
                if docID in top_docs:
                    term_scores[term] += tf * idf

    for t in orig_terms:
        term_scores.pop(t, None)
    expanded = [t for t,_ in sorted(term_scores.items(), key=lambda x: x[1], reverse=True)[:EXPAND_TERMS]]
    return orig_terms + expanded

def main():
    dict_file=postings_file=query_file=out_file=None
    try:
        opts,_ = getopt.getopt(sys.argv[1:], 'd:p:q:o:')
    except:
        usage(); sys.exit(2)
    for o,a in opts:
        if o=='-d': dict_file=a
        elif o=='-p': postings_file=a
        elif o=='-q': query_file=a
        elif o=='-o': out_file=a
    if not (dict_file and postings_file and query_file and out_file): usage(); sys.exit(2)

    dictionary = load_dictionary(dict_file)
    doc_lengths = load_doc_lengths(postings_file)
    total_docs = len({d for d,_ in doc_lengths if _=='content'})

    nltk.download('punkt', quiet=True)
    with open(query_file,'r',encoding='utf8') as qf:
        query = qf.readline().strip()

    orig_terms = preprocess(query)
    initial_scores = compute_scores(orig_terms, dictionary, postings_file, doc_lengths, total_docs)
    top_docs = [doc for doc,_ in sorted(initial_scores.items(), key=lambda x:x[1], reverse=True)[:TOP_K_DOCS]]

    all_terms = expand_query(orig_terms, top_docs, dictionary, postings_file, doc_lengths, total_docs)
    final_scores = compute_scores(all_terms, dictionary, postings_file, doc_lengths, total_docs)

    ranked = [doc for doc,_ in sorted(final_scores.items(), key=lambda x:x[1], reverse=True)]
    with open(out_file,'w',encoding='utf8') as outf:
        outf.write(' '.join(map(str,ranked)) + '\n')

if __name__=='__main__':
    main()
