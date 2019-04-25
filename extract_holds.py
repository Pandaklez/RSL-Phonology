# python 3.7.1
# Author: Anna Klezovich
# The script is based on Calle Börstell make_signs_still.py script for SSL and NGT
# https://github.com/borstell/make_sign_stills

# This script extracts holds for signs of RSL.

import cv2, os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import imutils
import cyrtranslit


def hist(img):
    """Returns a histogram analysis of a single image (in this case, a video frame)

    256 colours on the x axis and number of the pixels in each of these colours on the y axis
    """
    hist_image = cv2.calcHist([img], [0], None, [256], [0, 256])
    return hist_image

def occupied(frames):
    """Returns indices of occupied frames

    The first frame is the setting. When hands appear and move on the video, the frames becomes occupied
    Setting frames cannot be key frames with holds. This function cuts them out.

    This function is based on motion detection tutorial here
    (https://www.pyimagesearch.com/2015/05/25/basic-motion-detection-and-tracking-with-python-and-opencv/)
    """
    firstFrame = None

    occupied_frames_ind = []

    # loop over the frames of the video
    i = 0
    for frame in frames:
        # grab the current frame and initialize the occupied/unoccupied text
        text = "Unoccupied"

        # resize the frame, convert it to gray scale, and blur it
        frame = imutils.resize(frame, width=500)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        # if the first frame is None, initialize it
        if firstFrame is None:
            firstFrame = gray
            continue

        # compute the absolute difference between the current frame and first frame
        frameDelta = cv2.absdiff(firstFrame, gray)
        thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]

        # dilate the thresholded image to fill in holes,
        # then find contours on thresholded image
        thresh = cv2.dilate(thresh, None, iterations=2)
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        # loop over the contours
        for c in cnts:
            # if the contour is too small, ignore it
            if cv2.contourArea(c) < 9000.0:  # hardcoded on my data
                continue

            # No need here to compute the bounding box for the contour
            # So we just update the text
            text = "Occupied"

        if text == "Occupied":
            occupied_frames_ind.append(i)
        i += 1

    return occupied_frames_ind

def read_frames(video):
    """
    Reads a video file and returns a list of the histogram data only for occupied frames
    """
    v = cv2.VideoCapture(video)
    frames = []
    frames_hist = []
    success,image = v.read()
    while success:
        success,image = v.read()
        if success:
            frames.append(image)
            frames_hist.append(hist(image))

    # We exclude setting frames where there are no hands and no movement
    occupied_frames_ind = occupied(frames)

    occupied_frames_hist = [frames_hist[i] for i in occupied_frames_ind]

    # In some cases we are losing the last peak
    # Thus, we add 2 unoccupied frames in the end to to account for the last peak
    ending = frames_hist[(occupied_frames_ind[-1] + 1): (occupied_frames_ind[-1] + 3)]
    occupied_frames_hist.extend(ending)

    # Offset is a number of frames cut in the beginning of the video
    # We add offset to key frames later
    offset = occupied_frames_ind[0]

    return occupied_frames_hist, offset

def get_frame_difference(video):
    """
    Goes through the histograms of video frames pairwise and returns a list of
    frame indices (x) and histogram differences (y)
    """
    frames, offset = read_frames(video)
    x = []
    y = []
    for n,f in enumerate(frames):
        if n!=len(frames)-1:
            x.append(n)
            y.append(1-( cv2.compareHist(hist(f), hist(frames[n+1]), cv2.HISTCMP_BHATTACHARYYA) ) )  # HISTCMP_CORREL

    y = smooth(y, 3)
    return x,y, offset

def smooth(y, window):
    """
    Smoothes frames differences over frames with the help of the moving average
    """
    box = np.ones(window) / window
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def plot_changes(video):
    """
    Returns a plot of the frame histogram differences over video frames
    (NB: not necessary for the analysis)
    """
    plotname = video.split(".")[0]+"_plot_frames.png"
    x,y, offset = get_frame_difference(video)

    # y = [1 - n for n in y]

    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set(xlabel='Frame', ylabel='Difference', title='Frame differences over time')
    ax.grid()

    fig.savefig("bht_rev_" + plotname)
    # plt.show()

def get_key_frames(video):
    """
    Reads through the frame differences of a video,
    gets two highest peaks. When there is initially only one peak -> returns one peak
    """

    x,y, offset = get_frame_difference(video)
    y = [1 - n for n in y]

    x = np.array(x)
    y = np.array(y)

    peaks, properties = signal.find_peaks(y, height=0)
    peak_prop = {}
    for i, el in enumerate(peaks):
        key = (-1) * properties['peak_heights'][i]
        peak_prop[key] = el

    peak_prop = sorted(peak_prop.items())
    two_highest = peak_prop[:2]
    frames = [peak[1] for peak in two_highest]

    # When a very short video is smoothed, sometimes there are no peaks
    # If this happens we just take one frame from the middle as a peak
    if frames == []:
        frames.append(np.median(x))

    return frames, offset

def save_key_frames(video):
    """
    Saves the frames that are estimated holds as image files and returns a list of their names
    """
    outfile = video.split(".")[0]
    outfile = outfile.split("_")[1]
    outfile = cyrtranslit.to_latin(outfile, 'ru')
    all_frames, offset = get_key_frames(video)

    frames = [el+offset for el in all_frames]

    count = 1
    file_names = []
    for f in frames:
        v = cv2.VideoCapture(video)
        v.set(1,f-1)
        ret,frame = v.read()
        filename = outfile+"_frame" + str(count) + ".jpg"
        cv2.imwrite(filename, frame)
        file_names.append(filename)
        count += 1
    return file_names

def main():
    """
    Iterates over files in directory and saves images of key frames for each .mp4 file
    """
    print("Progress...")

    for f in os.listdir():
        if f.endswith(".mp4"):

            print(f.split(".")[0])

            #plot_changes(f)  # Uncomment if you want to create plots of changes in-between frames
            save_key_frames(f)

    print("Done")

if __name__=="__main__":
    main()



