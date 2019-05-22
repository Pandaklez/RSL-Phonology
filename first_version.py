# python 3.7.1
# Author: Anna Klezovich
# The script is based on Calle Börstell make_signs_still.py script for SSL and NGT
# https://github.com/borstell/make_sign_stills

# This script extracts holds for signs of RSL.
# (Check out the second version of this code for a better accuracy)


import cv2, os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import cyrtranslit

def hist(img):
    """
    Returns a histogram analysis of a single image (in this case, a video frame)
    """
    hist_image = cv2.calcHist([img], [0], None, [256], [0, 256])
    return hist_image

def read_frames(video):
    """
    Reads a video file and returns a list of the histogram data for each frame
    """
    print(video.split(".")[0])
    v = cv2.VideoCapture(video)
    frames = []
    frames_hist = []
    success,image = v.read()
    while success:
        success,image = v.read()
        if success:
            frames.append(image)
            frames_hist.append(hist(image))

    return frames_hist

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

    return x,y

def plot_changes(video):
    """
    Returns a plot of the frame histogram differences over video frames
    (NB: not necessary for the analysis)
    """
    plotname = video.split(".")[0]+"_plot_frames.png"
    x,y = get_frame_difference(video)

    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set(xlabel='Frame', ylabel='Difference', title='Frame differences over time')
    ax.grid()

    #fig.savefig(plotname)
    #plt.show()

def get_key_frames(video):
    """
    Reads through the frame differences of a video,
    gets standard deviation of peaks' prominences. If prominence is more than 3 std, than it is a hold estimate
    """

    x,y = get_frame_difference(video)
    y = [1 - n for n in y]

    x = np.array(x)
    y = np.array(y)

    diff = list(zip(y, x))

    peaks, _ = signal.find_peaks(y)
    first = peaks[0]
    y = y[first:]

    peaks, _ = signal.find_peaks(y)

    prominences = signal.peak_prominences(y, peaks)[0]
    std = np.std(prominences)
    key_prom = []
    for p in prominences:
        if p > 3*std:
            key_prom.append(p)

    min_p = np.min(key_prom) - 0.0000001

    key_peaks, _ = signal.find_peaks(y, prominence=min_p)

    frames = [peak+first for peak in key_peaks]
    return frames

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
    filenames = []
    for f in frames:
        v = cv2.VideoCapture(video)
        v.set(1,f-1)
        ret,frame = v.read()
        filename = outfile+"_frame"+str(count)+".jpg"
        cv2.imwrite(filename, frame)
        filenames.append(filename)
        count += 1
    return filenames

def main():
    """
    Iterates over files in directory and creates overlay images of key frames for each .mp4 file
    """
    for f in os.listdir():
        if f.endswith(".mp4"):
            #plot_changes(f)  # Uncomment if you want to create plots of changes in-between frames
            save_key_frames(f)

if __name__=="__main__":
    main()