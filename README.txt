This is the README file for A0315045N-A0245219E-A0312970H-A0312824L's submission
Email(s): e1508640@u.nus.edu, e0893443@u.nus.edu, e1496695@u.nus.edu, e1496620@u.nus.edu

== Python Version ==

I'm (We're) using Python Version 3.12.3 for this assignment.

== General Notes about this assignment ==

=== Index.py ===

I broke down the dataset into the fields (zone):
- Title
- Content
- Court
- Date

The dictionary.txt is in such a format:
zone:term [byte_offset][number_of_documents]

For the postings.txt it is in such a format:
At the top (Document Vector Lengths):
LC 1001 3.56 # Content vector length for doc 1001
LT 1001 2.12 # Title vector length for doc 1001
Followed by Postings Lists (per term):
doc_id:term_frequency doc_id:term_frequency

=== Search.py ===

The searching logic is based on standard (homework-3-style) tf×idf ranked retrieval. All
queries are converted to a format suitable for this - the special syntax used in boolean
retrieval is stripped away (ANDs are ignored, phrase queries are treated as independent
words).

All text is normalized into NFKD to neutralize Unicode representational differences, then
sent through nltk tokenization and Porter stemming.

The query is ran on the `title` and `content` zones of each document separately, allowing
us to give different weights depending on which zone a match is in - inspired by Web
search techniques. We find that giving heigher weight to the `title` improves performance,
although the information within in does not look useful for most case retrieval tasks.

=== Query Refinement ===

We have implemented two independent query refinement techniques:

==== WordNet‑Based Query Expansion (search_tfidf_weight_wordnet.py)

Method:
Look up each stemmed input term in WordNet synsets, select high-confidence synonyms,
and append them to the query before ranking.

==== Pseudo‑Relevance Feedback (PRF) (search_prf.py)

Method:
- Perform an initial retrieval with the baseline model.
- Take the top K=5 documents as relevant feedback.
- Score all content terms in those docs by tf·idf, pick top 10 new terms (not in the
  original query), and append them.
- Rerank with the expanded query using the same weighted TF×IDF.

Each technique is invoked by choosing its corresponding search script.
We carried out experiments comparing baseline, WordNet expansion, and PRF on the
development queries and report MAF2 scores in this README.

== Files included with this submission ==

README.txt                      This file
index.py                        Indexing program
search_prf.py                   WordNet-based searching program
search_tfidf_weight_wordnet.py  PRF-based searching program
search.py                       Searching program for evaluation (same as XXX FIXME XXX)
dictionary.txt                  Dictionary file of the index
postings.txt                    Postings file of the index
BONUS.docx                      For bonus marks qualificatiion

== Statement of individual work ==

Please put a "x" (without the double quotes) into the bracket of the appropriate statement.

[x] I/We, A0315045N-A0245219E-A0312970H-A0312824L, certify that I/we have followed the CS 3245 Information
Retrieval class guidelines for homework assignments.  In particular, I/we
expressly vow that I/we have followed the Facebook rule in discussing
with others in doing the assignment and did not take notes (digital or
printed) from the discussions.  

[ ] I/We, A0000000X, did not follow the class rules regarding homework
assignment, because of the following reason:

<Please fill in>

We suggest that we should be graded as follows:

<Please fill in>

== References ==

Normalization forms of Unicode
https://www.unicode.org/reports/tr15/
