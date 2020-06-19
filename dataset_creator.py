import cv2 as cv
import numpy as np
import time
import os
import hand_detection_functions as hd

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

def save_sample(dir, i):
    file_name = str(dir + '_' + str(i) + '.jpg')
    cv.imwrite(os.path.join(path, file_name), sil)
    return file_name

if __name__ == "__main__":
    i, path, dir = initialize_dataset()

    cap, term_crit = hd.initialize_video()

    r, h, c, w, track_window = hd.initialize_hand_window()

    frame, bg_roi, hand_roi = hd.position_hand(cap, c, r, w, h)

    hand_mask, diff = hd.subtract_background(bg_roi, hand_roi)

    roi_hsv, roi_hist = hd.get_histogram(hand_mask, hand_roi)

    while(1):
        ret, frame = cap.read()
        norm = cv.normalize(frame, frame, 0, 255, cv.NORM_MINMAX)
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        dst = cv.calcBackProject([hsv], [0,1], roi_hist, [0, 180, 0, 256], 1)

        # apply camshift to get the new location
        ret, track_window = cv.CamShift(dst, track_window, term_crit)

        # Draw it on image
        img_camshift, pts = hd.draw_camshift(frame, ret)

        # Get camshift extrema
        x,y,w,h = hd.get_camshift_extrema(pts)

        # Create hand window (larger box)
        large_x, large_y, large_w, large_h = hd.get_current_hand_window(x,y,w,h)

        # Draw the larger rectangle on frame
        draw_rect = cv.rectangle(img_camshift, (x - int(w / 2), y - int(h / 2)), (x + int(1.5 * w), y + h), (0, 255, 0), 1)

        cv.imshow("Display", draw_rect)

        if cv.waitKey(1) & 0xFF == ord('q'):
             break
        if cv.waitKey(1) & 0xFF == ord('c'):
            roi, roi_bgr = hd.get_ROI_image(frame, hsv, (x,y), (large_x, large_y), (large_w, large_h))
            sil = hd.get_mask(roi, roi_bgr, roi_hist)
            file_name = save_sample(dir,i)
            cv.imshow(file_name, sil)
            k = cv.waitKey(500)
            i = i+1

    cap.release()
    cv.destroyAllWindows()