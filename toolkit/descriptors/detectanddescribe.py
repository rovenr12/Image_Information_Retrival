import numpy as np

class DetectAndDescribe:
    def __init__(self, detector, descriptor):
        # Store the keypoint detector and local invariant descriptor
        self.detector = detector
        self.descriptor = descriptor

    def describe(self, image, useKpList=True):
        # Detect keypoints in the image and extract local invariant descriptors
        kps = self.detector.detect(image)
        (kps, descs) = self.descriptor.compute(image, kps)

        # If there are no keypoints or descriptors, return None
        if len(kps) == 0:
            return (None, None)

        # Check to see if the keypoints should be converted to a NumPy array
        if useKpList:
            kps = np.int0([kp.pt for kp in kps])

        # Return a tuple of the keypoints and descriptors
        return (kps, descs)