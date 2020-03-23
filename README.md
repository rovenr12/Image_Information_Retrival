# Image Information Retrieval

In order to do the content-based information Retrieval, It requires 5 steps:
1. Extracting the features from the documents(Images)
2. Building the dictionary(Codebook) using K-mean
3. Forming the Bag-of-Visual-Words(BOVW) which is the histgram format telling
how many features each documents have.
4. Building the Redis-index that key is feature and the values are images that
containing this feautre
5. Search

# How to execute the file
Requirement Package:
1. openCV-python, openCV-contrib-python
2. imutils
3. h5py

Requirement Software
1.Redis - Execute it by redis-cli ping


Step 1: Extracting features by 'index_features.py'.
Line 25, 26 can use to choose the feautres. In order to use SIFT and SURF,
openCV have to be built!! Otherwise, it can use binary-descriptor such as
BRISK, Fast and so on.
Example: python index_features.py -d ukbench -f BRISK_BRISK/features.hdf5

Step 2: Creating Codebook by 'cluster_features.py'
Example: python cluster_features.py -f BRISK_BRISK/features.hdf5
-c BRISK_BRISK/vocab.cpickle -k 1536 -p 0.25

Step 3: Forming the Bag-of-Visual-Words(BOVW) by 'extract_bovw.py'
Example:python extract_bovw.py -f BRISK_BRISK/features.hdf5
-c BRISK_BRISK/vocab.cpickle -b BRISK_BRISK/bovw.hdf5 -d BRISK_BRISK/idf.cpickle

Step 4: Building the Redis-index by 'build_redis_index.py'
Example:python build_redis_index.py -b BRISK_BRISK/bovw.hdf5

Step 5: Search by 'search.py'
Make sure the value uses in Line 29 and 30 is the same as step 1
python search.py -d ukbench -f BRISK_BRISK/features.hdf5 -b BRISK_BRISK/bovw.hdf5
	-c BRISK_BRISK/vocab.cpickle -r ukbench/relevant.json -q ukbench/ukbench00258.jpg

# Performance
