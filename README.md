I broke down the dataset into the fields (zone):

- Title
- Content
- Court
- Date

The dictionary file is in such a format:
zone:term [byte_offset][number_of_documents]

For the Postings File it is in such a format:
At the top (Document Vector Lengths):
LC 1001 3.56 # Content vector length for doc 1001
LT 1001 2.12 # Title vector length for doc 1001
Followed by Postings Lists (per term):
doc_id:term_frequency doc_id:term_frequency

PS: It took about 80 minutes for me to index the 700mb csv
