import cv2 as cv
import numpy as np
import time

def initialize_video():
    """
    :return: Video-capture object, cam-shift algorithm termination criteria
    """
    cap = cv.VideoCapture(0)
    term_crit = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1)
    return cap, term_crit

def initialize_hand_window():
    """
    :return: Coordinates of initial window location
    """
    r, h, c, w = 130, 420, 940, 300
    track_window = (c, r, w, h)
    return r, h, c, w, track_window

def get_mask(roi, roi_bgr, roi_hist):
    """
    :param roi: Mat object at region of interest in HSV
    :param roi_bgr: Mat object at region of interest in BGR
    :param roi_hist: Histogram of region of interest
    :return: Thresholded mask of hand/region of interest
    """
    roi_backproj = cv.calcBackProject([roi], [0, 1], roi_hist, [0, 180, 0, 256], 1)
    kernel = np.ones((11, 11), np.uint8)
    large_kernel = np.ones((13, 13), np.uint8)
    small_kernel = np.ones((3, 3), np.uint8)

    cv.filter2D(roi_backproj, -1, kernel, roi_backproj)
    ret, thresh = cv.threshold(roi_backproj, 150, 255, cv.THRESH_BINARY)
    thresh = cv.merge((thresh, thresh, thresh))
    res = cv.bitwise_and(roi_bgr, thresh)
    res = cv.erode(res, small_kernel, iterations=2)
    res = cv.cvtColor(res, cv.COLOR_BGR2GRAY)

    # Make silhouette using thresholding
    ret, mask = cv.threshold(res, 10, 255, cv.THRESH_BINARY)

    #Opening and closing to remove noise
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, large_kernel)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)

    #resize image for consistency
    mask_250 = cv.resize(mask, (200,200))
    return mask_250

def position_hand(cap, c, r, w, h):
    """
    :param cap: Video capture object
    :param c: Column index for initial window position
    :param r: Row index for initial window position
    :param w: Width of initial window
    :param h: Height of initial window
    :return: Most recent frame, background image of region of interest, and image of hand over
    background region of interest
    """
    t_init = time.perf_counter()
    t_passed = 0
    while(1):
        t_passed = time.perf_counter()-t_init
        ret, frame = cap.read()
        message_1 = "Position Hand in Box Below"
        message_2 = "Press 's' to Start"
        cv.putText(frame, message_1, (920, 115), cv.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 1, cv.LINE_AA)
        rect = cv.rectangle(frame.copy(), (c,r), (c+w, r+h), (255,255,255), 3)
        cv.putText(rect, message_2, (940, 570), cv.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 1, cv.LINE_AA)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
        if (t_passed < 1.5):
            cv.namedWindow('Display')
            cv.moveWindow('Display', 40, 30)
            cv.imshow('Display', frame)
            bg_roi = frame[r:r+h, c:c+w]
        elif (t_passed >= 1.5):
            cv.namedWindow('Display')
            cv.moveWindow('Display', 40, 30)
            cv.imshow('Display', rect)
            if cv.waitKey(1) & 0xFF == ord('s'):
                cv.namedWindow('Display')
                cv.moveWindow('Display', 40, 30)
                cv.imshow('Display', rect)
                hand_roi = frame[r:r+h, c:c+w]
                break
        else:
            break
    return frame, bg_roi, hand_roi

def subtract_background(bg_roi, hand_roi):
    """
    :param bg_roi: Background image of region of interest
    :param hand_roi: Image of hand over background region of interest
    :return: Thresholded mask of hand, and background subtracted image
    """
    diff = cv.absdiff(bg_roi, hand_roi)
    diff = cv.cvtColor(diff, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(diff, 25, 255, cv.THRESH_BINARY)
    return thresh, diff

def get_histogram(hand_mask, hand_roi):
    """
    :param hand_mask: Mask of hand obtained from background subtraction
    :param hand_roi: Image of hand in region of interest
    :return: HSV image of region of interest, and histogram over the region of interest
    """
    # Erode mask
    kernel = np.ones((5,5),np.uint8)
    mask = cv.erode(hand_mask, kernel, iterations = 1)

    roi_hsv = cv.cvtColor(hand_roi, cv.COLOR_BGR2HSV)
    roi_hist = cv.calcHist([roi_hsv], [0,1], mask, [180,256], [0,180,0,256])
    cv.normalize(roi_hist, roi_hist, 0, 255, cv.NORM_MINMAX)
    return(roi_hsv, roi_hist)

def get_current_hand_window(x,y,w,h):
    """
    :param x: Current x index in video
    :param y: Current y index in video
    :param w: Current width of ROI in video
    :param h: Current height of ROI in video
    :return: Enlarged dimensions for the ROI window
    """
    x = x - int(w / 2)
    y = y - int(h / 2)
    w = int(1.5 * w)
    return x,y,w,h

def get_ROI_image(frame, hsv, pt, large_pt, large_dim):
    """
    :param frame: Current frame of video
    :param hsv: HSV frame of video
    :param pt: X and Y coordinates of region of interest
    :param large_pt: Englarged x and y coordinates of region of interest
    :param large_dim: Enlarged w and h dimensions of region of interest
    :return: Region of interest image in BGR and HSV
    """
    (x,y) = pt
    (large_x, large_y) = large_pt
    (large_w, large_h) = large_dim
    if (y+large_h >= frame.shape[0]) and (x+large_w >= frame.shape[1]):
        roi_bgr = frame[large_y:frame.shape[0], large_x:frame.shape[1]]
        roi = hsv[large_y:frame.shape[0], large_x:frame.shape[1]]
    elif y + large_h >= frame.shape[0]:
        roi_bgr = frame[large_y:frame.shape[0], large_x:x + large_w]
        roi = hsv[large_y:frame.shape[0], large_x:x + large_w]
    elif large_y <= 0:
        roi_bgr = frame[0:large_h, large_x:x+large_w]
        roi = hsv[0:large_h, large_x:x+large_w]
    elif x + large_w >= frame.shape[1]:
        roi_bgr = frame[large_y:y + large_h, large_x:frame.shape[1]]
        roi = hsv[large_y:y + large_h, large_x:frame.shape[1]]
    else:
        roi_bgr = frame[large_y:y + large_h, large_x:x + large_w]
        roi = hsv[large_y:y + large_h, large_x:x + large_w]
    return roi, roi_bgr

def get_camshift_extrema(pts):
    """
    :param pts: Set of points determined by camshift algorithm
    :return: Coordinates and dimensions of camshift extreme points
    """
    pts_col = pts[:, 0]
    pts_row = pts[:, 1]
    x = np.amin(pts_col)
    y = np.amin(pts_row)
    w = np.amax(pts_col) - x
    h = np.amax(pts_row) - y
    return x, y, w, h

def draw_camshift(frame, ret):
    """
    :param frame: Current frame of video
    :param ret: Points obtained from camshift alogrithm
    :return: Frame with camshift box added, and points defining box
    """
    pts = cv.boxPoints(ret)
    pts = np.int0(pts)
    img = cv.polylines(frame.copy(), [pts], True, 255, 2)
    return img, pts