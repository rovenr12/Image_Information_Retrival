from __future__ import print_function
from sklearn.cluster import MiniBatchKMeans
import numpy as np
import datetime
import h5py


class Vocabulary:
    def __init__(self, dbPath, verbose=True):
        # Store the database path and the verbosity setting
        self.dbPath = dbPath
        self.verbose = verbose

    def fit(self, numClusters, samplePercent, randomState=None):
        # Open the database and grab the total number of features
        db = h5py.File(self.dbPath)
        totalFeatures = db["features"].shape[0]

        # Determine the number of features to sample, generate the indexes of the sample,
        # sorting them in ascending order to speedup access time from the HDF5 database
        sampleSize = int(np.ceil(samplePercent * totalFeatures))
        idxs = np.random.choice(np.arange(0, totalFeatures), (sampleSize), replace=False)
        idxs.sort()
        data = []
        self._debug("starting sampling...")

        # Loop over the randomly sampled indexes and accumulate the features to cluster
        for i in idxs:
            data.append(db["features"][i][2:])
        # Cluster the features
        self._debug("sampled {:,} features from a population of {:,}".format(
            len(idxs), totalFeatures))
        self._debug("clustering with k={:,}".format(numClusters))
        clt = MiniBatchKMeans(n_clusters=numClusters, random_state=randomState)
        clt.fit(data)
        self._debug("cluster shape: {}".format(clt.cluster_centers_.shape))

        # Close the database
        db.close()

        # Return the cluster centroids
        return clt.cluster_centers_

    def _debug(self, msg, msgType="[INFO]"):
        # Check to see the message should be printed
        if self.verbose:
            print("{} {} - {}".format(msgType, msg, datetime.datetime.now()))