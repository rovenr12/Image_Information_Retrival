import numpy as np


class RedisQueue:
    def __init__(self, redisDB):
        # Store the redis database object
        self.redisDB = redisDB

    def add(self, imageIdx, hist):
        # Initialize the redis pipeline
        p = self.redisDB.pipeline()

        # Loop over all non-zero entries for the histogram, creating a
        # (visual word -> document) record for each visual word in the histogram
        for i in np.where(hist > 0)[0]:
            p.rpush("vw:{}".format(i), imageIdx)

        # Execute the pipeline
        p.execute()

    def finish(self):
        # Save the state of the Redis database
        self.redisDB.save()