from __future__ import print_function
from pyimagesearch.db import RedisQueue
from redis import Redis
import argparse
import h5py

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-b", "--bovw-db", required=True, help="Path to where the bag-of-visual-words database")
args = vars(ap.parse_args())

# Connect to redis, initialize the redis queue, and open the bag-of-visual-words database
redisDB = Redis(host="localhost", port=6379, db=0)
rq = RedisQueue(redisDB)
bovwDB = h5py.File(args["bovw_db"], mode="r")

# Loop over the entries in the bag-of-visual-words
for (i, hist) in enumerate(bovwDB["bovw"]):
    # Check to see if progress should be displayed
    if i > 0 and i % 10 == 0:
        print("[PROGRESS] processed {} entries".format(i))

    # Add the image index and histogram to the redis server
    rq.add(i, hist)

# Close the bag-of-visual-words database and finish the indexing processing
bovwDB.close()
rq.finish()