import cv2 as cv
import numpy as np
import time

def initialize_video():
    # Start video capturing
    cap = cv.VideoCapture(0)

    term_crit = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1)
    return cap, term_crit

def initialize_hand_window():
    # Initial window location
    r, h, c, w = 130, 420, 940, 300
    track_window = (c, r, w, h)
    return r, h, c, w, track_window

def get_mask(roi, roi_bgr, roi_hist):
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
    #Opening and closing to get rid of noise
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, large_kernel)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)

    #resize image for consistency
    mask_250 = cv.resize(mask, (200,200))
    return mask_250

# Display "Position Hand in Box Below" - 1.5s
# Take bg_roi after 1s
# Display Box below with "Press 's' to start" underneath
# If q_ord == s, and t>1.5s, take hand_roi and break

def position_hand(cap, c, r, w, h):
    t_init = time.perf_counter()
    t_passed = 0
    while(1):
        t_passed = time.perf_counter()-t_init
        ret, frame = cap.read()
        message_1 = "Position Hand in Box Below"
        message_2 = "Press 's' to Start"
        cv.putText(frame, message_1, (920, 115), cv.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 1, cv.LINE_AA)
        rect = cv.rectangle(frame, (c,r), (c+w, r+h), (255,255,255), 3)
        cv.putText(rect, message_2, (940, 570), cv.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 1, cv.LINE_AA)
        #942, 567
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
        if (t_passed < 1.5):
            cv.namedWindow('Display')
            cv.moveWindow('Display', 40, 30)
            cv.imshow('Display', frame)
            bg_roi = frame[r:r+h, c:c+w]
        elif (t_passed > 1.5):
            cv.namedWindow('Display')
            cv.moveWindow('Display', 40, 30)
            cv.imshow('Display', rect)
            if cv.waitKey(1) & 0xFF == ord('s'):
                cv.namedWindow('Display')
                cv.moveWindow('Display', 40, 30)
                cv.imshow('Display', rect)
                hand_roi = frame[r:r+h, c:c+w]
                break
        # elif (t_passed > 1) & (t_passed < 5.):
        #     cv.namedWindow('Display')
        #     cv.moveWindow('Display', 40, 30)
        #     cv.imshow('Display', rect)
        #     hand_roi = frame[r:r+h, c:c+w]
        else:
            break
    # Set up ROI for tracking
    return frame, bg_roi, hand_roi

def subtract_background(bg_roi, hand_roi):
    diff = cv.absdiff(bg_roi, hand_roi)
    diff = cv.cvtColor(diff, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(diff, 25, 255, cv.THRESH_BINARY)
    return thresh, diff

def get_histogram(hand_mask, hand_roi):
    # # Erode mask
    kernel = np.ones((5,5),np.uint8)
    mask = cv.erode(hand_mask, kernel, iterations = 1)
    # cv.imshow("Eroded", mask)
    # k = cv.waitKey(0)

    # # Get Histogram
    roi_hsv = cv.cvtColor(hand_roi, cv.COLOR_BGR2HSV)
    roi_hist = cv.calcHist([roi_hsv], [0,1], mask, [180,256], [0,180,0,256])
    cv.normalize(roi_hist, roi_hist, 0, 255, cv.NORM_MINMAX)
    return(roi_hsv, roi_hist)

def get_current_hand_window(x,y,w,h):
    x = x - int(w / 2)
    y = y - int(h / 2)
    w = int(1.5 * w)
    return x,y,w,h

def get_ROI_image(frame, hsv, pt, large_pt, large_dim):
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
    pts_col = pts[:, 0]
    pts_row = pts[:, 1]
    x = np.amin(pts_col)
    y = np.amin(pts_row)
    w = np.amax(pts_col) - x
    h = np.amax(pts_row) - y
    return x, y, w, h

def draw_camshift(frame, ret):
    pts = cv.boxPoints(ret)
    pts = np.int0(pts)
    img = cv.polylines(frame.copy(), [pts], True, 255, 2)
    return img, pts