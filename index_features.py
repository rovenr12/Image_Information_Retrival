from __future__ import print_function
from toolkit.descriptors import DetectAndDescribe
from toolkit.indexer import FeatureIndexer
from imutils.feature import FeatureDetector_create, DescriptorExtractor_create
from imutils import paths
import argparse
import imutils
import cv2
import time

start = time.time()
# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
                help="Path to the directory that contains the images to be indexed")
ap.add_argument("-f", "--features-db", required=True,
                help="Path to where the features database will be stored")
ap.add_argument("-a", "--approx-images", type=int, default=500,
                help="Approximate # of images in the dataset")
ap.add_argument("-b", "--max-buffer-size", type=int, default=50000,
                help="Maximum buffer size for # of features to be stored in memory")
args = vars(ap.parse_args())

# Initialize the keypoint detector, local invariant descriptor, and the descriptor pipeline
detector = FeatureDetector_create("BRISK")
descriptor = DescriptorExtractor_create("BRISK")
dad = DetectAndDescribe(detector, descriptor)

# Initialize the feature indexer
fi = FeatureIndexer(args["features_db"], estNumImages=args["approx_images"],
                    maxBufferSize=args["max_buffer_size"], verbose=True)
# Loop over the images in the dataset
for (i, imagePath) in enumerate(sorted(paths.list_images(args["dataset"]))):
    # Check to see if progress should be displayed
    if i > 0 and i % 10 == 0:
        fi._debug("processed {} images".format(i), msgType="[PROGRESS]")

    # Extract the image filename (i.e. the unique image ID) from the image path, then load the image itself
    filename = imagePath[imagePath.rfind("\\") + 1:]
    image = cv2.imread(imagePath)
    image = imutils.resize(image, width=320)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Get image descriptors
    (kps, descs) = dad.describe(image)

    # If either the keypoints or descriptors are None, then ignore the image
    if kps is None or descs is None:
        continue

    # Index the features
    fi.add(filename, kps, descs)

# Finish the indexing process
fi.finish()
end = time.time()
print(end - start)