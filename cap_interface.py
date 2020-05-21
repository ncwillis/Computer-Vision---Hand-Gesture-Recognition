import cv2 as cv
import numpy as np
import time

def normalize_color(img):
    b,g,r = img[:,:,0], img[:,:,1], img[:,:,2]
    sum = b+g+r
    norm = img.copy()
    norm[:, :, 0] = np.uint8((b / sum) * 255)
    norm[:, :, 1] = np.uint8((g / sum) * 255)
    norm[:, :, 2] = np.uint8((r / sum) * 255)
    norm_rgb = cv.convertScaleAbs(norm)
    return norm_rgb

def increase_contrast(img):
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    equ = cv.equalizeHist(img)
    # clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # cl1 = clahe.apply(img)
    return equ

cap = cv.VideoCapture(0)

# Use camshift algorithm to determine location of hand
# Need to locate hand to have box around it
# Use grabcut algorithm to extract foreground (hand)
# Maybe start with a box showing where to put your hand, followed and use the camshift algorithm to make sure the box is
# always keeping track of the hand
# Can have the box stay still until a certain number of significant contours are present, then the camshift algorithm will start

# MEANSHIFT INITIALIZATION

# Initial window location
r,h,c,w = 130,420,940,300
track_window = (c,r,w,h)

def position_hand():
    t_init = time.perf_counter()
    t_passed = 0
    while(1):
        ret, frame = cap.read()
        frame = cv.rectangle(frame, (c,r), (c+w, r+h), (0,0,255), 3)
        cv.imshow('Display', frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
        if (time.perf_counter()-t_init) > 5.:
            break
    # Set up ROI for tracking
    ret, frame = cap.read()
    roi = frame[r:r + h, c:c + w]
    return frame, roi

frame, roi = position_hand()


# norm = normalize_color(roi)
# # res = np.hstack((roi, norm))
# cv.imshow("normalised", res)
# k = cv.waitKey(0)

# Add edges for easier detection for grabcut algo
contrast = increase_contrast(roi)
cv.imshow("Contrast", contrast)
k = cv.waitKey(0)

# roi_gray = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
roi_gray = cv.GaussianBlur(contrast, (11,11), 0)
edges = cv.Canny(roi_gray, 40, 100)
cnts, hrch = cv.findContours(edges.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
draw = cv.drawContours(roi.copy(), cnts, -1, (255, 255,255), 2)
cv.imshow("edges", draw)
k = cv.waitKey(0)

# exctract foreground
# roi_grabcut = cv.cvtColor(draw.copy(), cv.COLOR_GRAY2BGR)
roi_grabcut = draw.copy()
mask_grabcut = np.zeros(roi_grabcut.shape[:2],np.uint8)
bgdModel = np.zeros((1,65),np.float64)
fgdModel = np.zeros((1,65),np.float64)
rect = (0,0,w-1,h-1)
cv.grabCut(roi_grabcut,mask_grabcut,rect,bgdModel,fgdModel,5,cv.GC_INIT_WITH_RECT)
cv.grabCut(roi_grabcut,mask_grabcut,rect,bgdModel,fgdModel,5,cv.GC_INIT_WITH_RECT)
mask2 = np.where((mask_grabcut==2)|(mask_grabcut==0),0,1).astype('uint8')
roi_grabcut = roi_grabcut*mask2[:,:,np.newaxis]
cv.imshow("grabcut",roi_grabcut)
k = cv.waitKey(0)

# Make mask
ret, mask = cv.threshold(roi_grabcut, 20, 255, cv.THRESH_BINARY)
mask = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)
cv.imshow("mask", mask)
k = cv.waitKey(0)

# Erode mask
kernel = np.ones((5,5),np.uint8)
mask = cv.erode(mask, kernel, iterations = 5)
cv.imshow("Eroded", mask)
k = cv.waitKey(0)

# Get Histogram
roi_hsv = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
roi_hist = cv.calcHist([roi_hsv], [0,1], mask, [180,256], [0,180,0,256])
cv.normalize(roi_hist, roi_hist, 0, 255, cv.NORM_MINMAX)

while(1):
    ret, frame = cap.read()
    frame_hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    back_proj = cv.calcBackProject([frame_hsv], [0,1], roi_hist, [0, 180,0,256], 1)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (25, 25))
    cv.filter2D(back_proj, -1, kernel, back_proj)
    ret, thresh = cv.threshold(back_proj, 150, 255, cv.THRESH_BINARY)
    thresh = cv.merge((thresh, thresh, thresh))
    res = cv.bitwise_and(frame, thresh)

    cv.imshow("Frame", res)
    if cv.waitKey(1) & 0xFF == ord('q'):
         break


# Convert to HSV

# roi_hsv = cv.cvtColor(roi_grabcut, cv.COLOR_BGR2HSV)
# roi_gray = cv.cvtColor(roi_grabcut, cv.COLOR_BGR2GRAY)
# ret, mask = cv.threshold(roi_gray, 50, 255, cv.THRESH_BINARY)
# mask = cv.inRange(roi_hsv, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
# cv.imshow("Mask", mask)
# k = cv.waitKey(0)
# ch = (0, 0)
# hue = np.empty(roi_hsv.shape, roi_hsv.dtype)
# cv.mixChannels([roi_hsv], [hue], ch)
# # roi_hist = cv.calcHist([hue],[0],None,[180],[0,180])
# roi_hist = cv.calcHist([roi], [0], mask, [256], [0,256])
# cv.normalize(roi_hist,roi_hist,0,255,cv.NORM_MINMAX)

# termination criteria
term_crit = ( cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1 )

img_cap = False
# while(1):
#     ret, frame = cap.read()
#     norm = cv.normalize(frame, frame, 0, 255, cv.NORM_MINMAX)
#     hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
#     ch = (0, 0)
#     hue = np.empty(hsv.shape, hsv.dtype)
#     cv.mixChannels([hsv], [hue], ch)
#     dst = cv.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
#
#     # blank = np.ones(frame.shape[:2])
#     # frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
#     # frame_gray = cv.GaussianBlur(frame_gray, (21, 21), 0)
#     # edges = cv.Canny(frame_gray, 10, 60)
#     # cnts, hrch = cv.findContours(edges.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
#     # draw = cv.drawContours(blank, cnts, -1, (0, 255, 0), 2)
#
#     # apply meanshift to get the new location
#     ret, track_window = cv.CamShift(dst, track_window, term_crit)
#
#     # Draw it on image
#     pts = cv.boxPoints(ret)
#     pts = np.int0(pts)
#     img2 = cv.polylines(frame, [pts], True, 255, 2)
#
#     cv.imshow('frame', norm)
#
#     if cv.waitKey(1) & 0xFF == ord('q'):
#          break
#     if cv.waitKey(1) & 0xFF == ord('c'):
#         img_cap = True
#         break

cap.release()
cv.destroyAllWindows()

if img_cap == True:
    cv.imshow("Capture", frame)
    k = cv.waitKey(0)