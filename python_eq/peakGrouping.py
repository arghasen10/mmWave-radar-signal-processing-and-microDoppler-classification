import numpy as np

def peakGrouping(detMat):
    numDetectedObjects = detMat.shape[1]
    objOut = np.empty((3, 0))

    # Sort the detMat matrix according to the cell power
    order = np.argsort(detMat[2, :])[::-1]
    detMat = detMat[:, order]

    for ni in range(numDetectedObjects):
        detectedObjFlag = 1
        rangeIdx = int(detMat[1, ni])
        dopplerIdx = int(detMat[0, ni])
        peakVal = detMat[2, ni]
        kernal = np.zeros((3, 3))

        # Fill the middle column of the kernel
        kernal[1, 1] = peakVal

        need_index = np.where((detMat[0, :] == dopplerIdx) & (detMat[1, :] == rangeIdx + 1))[0]
        if need_index.size > 0:
            kernal[0, 1] = detMat[2, need_index[0]]

        need_index = np.where((detMat[0, :] == dopplerIdx) & (detMat[1, :] == rangeIdx - 1))[0]
        if need_index.size > 0:
            kernal[2, 1] = detMat[2, need_index[0]]

        # Fill the left column of the kernel
        need_index = np.where((detMat[0, :] == dopplerIdx - 1) & (detMat[1, :] == rangeIdx + 1))[0]
        if need_index.size > 0:
            kernal[0, 0] = detMat[2, need_index[0]]

        need_index = np.where((detMat[0, :] == dopplerIdx - 1) & (detMat[1, :] == rangeIdx))[0]
        if need_index.size > 0:
            kernal[1, 0] = detMat[2, need_index[0]]

        need_index = np.where((detMat[0, :] == dopplerIdx - 1) & (detMat[1, :] == rangeIdx - 1))[0]
        if need_index.size > 0:
            kernal[2, 0] = detMat[2, need_index[0]]

        # Fill the right column of the kernel
        need_index = np.where((detMat[0, :] == dopplerIdx + 1) & (detMat[1, :] == rangeIdx + 1))[0]
        if need_index.size > 0:
            kernal[0, 2] = detMat[2, need_index[0]]

        need_index = np.where((detMat[0, :] == dopplerIdx + 1) & (detMat[1, :] == rangeIdx))[0]
        if need_index.size > 0:
            kernal[1, 2] = detMat[2, need_index[0]]

        need_index = np.where((detMat[0, :] == dopplerIdx + 1) & (detMat[1, :] == rangeIdx - 1))[0]
        if need_index.size > 0:
            kernal[2, 2] = detMat[2, need_index[0]]

        # Compare the detected object to its neighbors. Detected object is at index [1, 1]
        if kernal[1, 1] != np.max(kernal):
            detectedObjFlag = 0

        if detectedObjFlag == 1:
            objOut = np.hstack((objOut, detMat[:, ni].reshape(-1, 1)))

    return objOut
