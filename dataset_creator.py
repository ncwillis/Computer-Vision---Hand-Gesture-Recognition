import cv2 as cv
import numpy as np
import time
import os

def get_silhouette(roi, roi_bgr):
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

def position_hand():
    t_init = time.perf_counter()
    t_passed = 0
    while(1):
        t_passed = time.perf_counter()-t_init
        ret, frame = cap.read()
        rect = cv.rectangle(frame, (c,r), (c+w, r+h), (0,0,255), 3)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
        if (t_passed < 0.5):
            cv.imshow('Display', frame)
            bg_roi = frame[r:r+h, c:c+w]
        elif (t_passed > 0.5) & (t_passed < 5.):
            cv.imshow('Display', rect)
            hand_roi = frame[r:r+h, c:c+w]
        else:
            break
    # Set up ROI for tracking
    return frame, bg_roi, hand_roi

def subtract_background(bg_roi, hand_roi):
    diff = cv.absdiff(bg_roi, hand_roi)
    diff = cv.cvtColor(diff, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(diff, 25, 255, cv.THRESH_BINARY)
    return thresh, diff

def get_histogram(hand_mask):
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

def initialize_dataset():
    # Create dataset directory to store files
    print("Enter Dataset Name and press Enter: ")
    dir = input()

    cwd = os.getcwd()
    path = os.path.join(cwd, "datasets")
    if "test" in dir:
        path = os.path.join(path, 'testing')
    else:
        path = os.path.join(path, 'training')
    path = os.path.join(path, dir)
    if not os.path.exists(path):
        os.mkdir(path)
    i = len(os.listdir(path)) + 1
    return i, path, dir

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

def draw_camshift(frame, ret):
    pts = cv.boxPoints(ret)
    pts = np.int0(pts)
    img = cv.polylines(frame.copy(), [pts], True, 255, 2)
    return img, pts

def get_camshift_extrema(pts):
    pts_col = pts[:, 0]
    pts_row = pts[:, 1]
    x = np.amin(pts_col)
    y = np.amin(pts_row)
    w = np.amax(pts_col) - x
    h = np.amax(pts_row) - y
    return x, y, w, h

def get_current_hand_window(x,y,w,h):
    x = x - int(w / 2)
    y = y - int(h / 2)
    w = int(1.5 * w)
    return x,y,w,h

def get_ROI_image(frame, hsv):
    if y + large_h >= frame.shape[0]:
        roi_bgr = frame[large_y:frame.shape[0], large_x:x + large_w]
        roi = hsv[large_y:frame.shape[0], large_x:x + large_w]
    if x + large_w >= frame.shape[1]:
        roi_bgr = frame[large_y:y + large_h, large_x:frame.shape[1]]
        roi = hsv[large_y:y + large_h, large_x:frame.shape[1]]
    else:
        roi_bgr = frame[large_y:y + large_h, large_x:x + large_w]
        roi = hsv[large_y:y + large_h, large_x:x + large_w]
    return roi, roi_bgr

def save_sample(dir, i):
    file_name = str(dir + '_' + str(i) + '.jpg')
    cv.imwrite(os.path.join(path, file_name), sil)
    return file_name

if __name__ == "__main__":
    i, path, dir = initialize_dataset()

    cap, term_crit = initialize_video()

    r, h, c, w, track_window = initialize_hand_window()

    frame, bg_roi, hand_roi = position_hand()

    hand_mask, diff = subtract_background(bg_roi, hand_roi)

    roi_hsv, roi_hist = get_histogram(hand_mask)

    while(1):
        ret, frame = cap.read()
        norm = cv.normalize(frame, frame, 0, 255, cv.NORM_MINMAX)
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        dst = cv.calcBackProject([hsv], [0,1], roi_hist, [0, 180, 0, 256], 1)

        # apply camshift to get the new location
        ret, track_window = cv.CamShift(dst, track_window, term_crit)

        # Draw it on image
        img_camshift, pts = draw_camshift(frame, ret)

        # Get camshift extrema
        x,y,w,h = get_camshift_extrema(pts)

        # Create hand window (larger box)
        large_x, large_y, large_w, large_h = get_current_hand_window(x,y,w,h)

        # Draw the larger rectangle on frame
        draw_rect = cv.rectangle(img_camshift, (x - int(w / 2), y - int(h / 2)), (x + int(1.5 * w), y + h), (0, 255, 0), 1)

        cv.imshow("Display", draw_rect)

        if cv.waitKey(1) & 0xFF == ord('q'):
             break
        if cv.waitKey(1) & 0xFF == ord('c'):
            roi, roi_bgr = get_ROI_image(frame, hsv)
            sil = get_silhouette(roi, roi_bgr)
            file_name = save_sample(dir,i)
            cv.imshow(file_name, sil)
            k = cv.waitKey(500)
            i = i+1

    cap.release()
    cv.destroyAllWindows()