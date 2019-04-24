import cv2, sys, math, os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import imutils
import datetime
from PIL import Image
import cyrtranslit

def hist(img):
    """
    Returns a histogram analysis of a single image (in this case, a video frame)
    """
    hist_image = cv2.calcHist([img], [0], None, [256], [0, 256])
    return hist_image  # cv2.normalize(hist_image, hist_image, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

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

    occup_frames_indeces = occupied(frames)
    print("all frames length", len(frames))
    print("indices ", occup_frames_indeces)
    frames = [frames[i] for i in occup_frames_indeces]

    occup_frames_hist = [frames_hist[i] for i in occup_frames_indeces]

    # we are losing last peak so let's add 2 more frames to make the second half of the last peak
    ending = frames_hist[(occup_frames_indeces[-1] + 1): (occup_frames_indeces[-1] + 3)]
    occup_frames_hist.extend(ending)
    print("occupied length", len(occup_frames_hist))

    offset = occup_frames_indeces[0]

    return occup_frames_hist, offset

def occupied(frames):
    firstFrame = None

    occupied_frames_ind = []

    # loop over the frames of the video
    i = 0
    for frame in frames:
        # grab the current frame and initialize the occupied/unoccupied
        # text
        text = "Unoccupied"

        # resize the frame, convert it to grayscale, and blur it
        frame = imutils.resize(frame, width=500)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        # if the first frame is None, initialize it
        if firstFrame is None:
            firstFrame = gray
            continue

        # compute the absolute difference between the current frame and
        # first frame
        frameDelta = cv2.absdiff(firstFrame, gray)
        thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]

        # dilate the thresholded image to fill in holes, then find contours
        # on thresholded image
        thresh = cv2.dilate(thresh, None, iterations=2)
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        # loop over the contours
        for c in cnts:
            # print(cv2.contourArea(c))
            # if the contour is too small, ignore it
            if cv2.contourArea(c) < 9000.0:
                continue

            # compute the bounding box for the contour, draw it on the frame,
            # and update the text
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            text = "Occupied"

        if text == "Occupied":
            occupied_frames_ind.append(i)
        i += 1

    return occupied_frames_ind

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

    #y = [1 - n for n in y]

    print(plotname)
    #print("x: ", x)
    #print("y: ", y)

    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set(xlabel='Frame', ylabel='Difference', title='Frame differences over time')
    ax.grid()

    fig.savefig("bht_rev_" + plotname)
    #plt.show()

def get_key_frames_GREAT(video):  # FIRST VARIANT OF MY ALGO widths are better without setting, prominence with setting
    """
    Reads through the frame differences of a video,
    gets standard deviation of peaks' prominences. If prominence is more than 3 std, than it is a hold estimate
    """

    x,y, offset = get_frame_difference(video)
    y = [1 - n for n in y]
    print("x = ", x)
    print("y = ", y)

    x = np.array(x)
    y = np.array(y)

    diff = list(zip(y, x))

    #peaks, _ = signal.find_peaks(y)
    #first = peaks[0]
    #y = y[first:]

    peaks, _ = signal.find_peaks(y)

    #prominences = signal.peak_prominences(y, peaks)[0]
    #std = np.std(prominences)
    #key_prom = []
    #for p in prominences:
    #    if p > 2*std:
    #        key_prom.append(p)

    #print(key_prom)
    # print(prominences)

    widths = signal.peak_widths(y, peaks)[0]
    std = np.std(widths)
    key_prom = []
    for p in widths:
        if p > 2 * std:
            key_prom.append(p)

    min_p = np.min(key_prom) - 0.0000001

    #key_peaks, _ = signal.find_peaks(y, prominence=min_p)
    key_peaks, _ = signal.find_peaks(y, width=min_p)

    #frames = [peak+first for peak in key_peaks]
    frames = [peak for peak in key_peaks]
    return frames, offset

def get_key_frames(video):
    """
    Reads through the frame differences of a video,
    gets two highest peaks. When there is initially only one peak -> returns one peak
    """

    x,y, offset = get_frame_difference(video)
    y = [1 - n for n in y]
    print("x = ", x)
    print("y = ", y)

    x = np.array(x)
    y = np.array(y)

    diff = list(zip(y, x))

    peaks, properties = signal.find_peaks(y, height=0)
    peak_prop = {}
    for i, el in enumerate(peaks):
        key = (-1) * properties['peak_heights'][i]
        peak_prop[key] = el

    peak_prop = sorted(peak_prop.items())
    two_highest = peak_prop[:2]
    frames = [peak[1] for peak in two_highest]

    return frames, offset

def get_key_frames_0(video):
    """
    Reads through the frame differences of a video, assumes the first peak to be the start of the sign,
    then returns the negative peaks (i.e. estimated holds) of the remaining frames
    """
    x,y = get_frame_difference(video)
    # print("x", x)
    # print("y", y)
    #y = [1 - n for n in y]

    diff = list(zip(y, x))

    #peaks, p = signal.find_peaks(y)
    #first = peaks[0]
    #y = y[first:]
    #peaks2, p = signal.find_peaks(y, prominence=0.07)

    # CORREL
    # peaks = signal.find_peaks_cwt(y,np.arange(1,15)) # 1,15 These are hardcoded figures, you may need to adjust (e.g. 15,25)
    # first = peaks[0] #0
    # neg = [1-n for n in y[first:]]
    # peaks2 = signal.find_peaks_cwt(neg,np.arange(1.5,8)) # 1.5,8 These are hardcoded figures, you may need to adjust (e.g. 15,45)
    # frames = [peak+first for peak in peaks2]

    # BHATTACHARYYA
    neg = [1 - n for n in y]
    peaks = signal.find_peaks_cwt(neg, np.arange(1, 15))
    first = peaks[1]   # так получилось комету сделать # 0
    frames = [peak+first for peak in peaks]

    #frames = [peak+first for peak in peaks2]
    #print("frames:", frames)
    return frames


def save_key_frames(video):
    """
    Saves the frames that are estimated holds as image files and returns a list of their names
    (NB: only frames in the first half of the list of key frames are saved, as later frames are assumed
    to constitute final rest position)
    """
    outfile = video.split(".")[0]
    outfile = outfile.split("_")[1]
    outfile = cyrtranslit.to_latin(outfile, 'ru')
    all_frames, offset = get_key_frames(video)

    frames = [el+offset for el in all_frames] # Uncomment if you want all key frames to be included
    print(frames)
    #frames = all_frames[:math.ceil(len(all_frames)/2)] # Comment out if you want all key frames included
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

def make_images(video):
    """
    Creates overlay images of relevant key frames generated from videos and deletes individual frames
    """
    imgs = save_key_frames(video)
    return imgs

def main():
    """
    Iterates over files in directory and creates overlay images of key frames for each .mp4 file
    """
    for f in os.listdir():
        if f.endswith(".mp4"):
            plot_changes(f)  # Uncomment if you want to create plots of changes in-between frames
            make_images(f)
    #video = "video_солнце_ru.ru0.mp4"
    #make_images(video)

if __name__=="__main__":
    main()