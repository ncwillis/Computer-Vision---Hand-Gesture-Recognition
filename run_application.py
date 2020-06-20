import cv2 as cv
import tensorflow as tf
import helper_functions.hand_detection_functions as hd
import helper_functions.classification_functions as cf

if __name__ == "__main__":
    cap, term_crit = hd.initialize_video()

    r, h, c, w, track_window = hd.initialize_hand_window()

    frame, bg_roi, hand_roi = hd.position_hand(cap, c, r, w, h)

    hand_mask, diff = hd.subtract_background(bg_roi, hand_roi)

    roi_hsv, roi_hist = hd.get_histogram(hand_mask, hand_roi)

    cnn = tf.keras.models.load_model('cnn.h5')

    while(1):
        ret, frame = cap.read()
        norm = cv.normalize(frame, frame, 0, 255, cv.NORM_MINMAX)
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        dst = cv.calcBackProject([hsv], [0,1], roi_hist, [0, 180, 0, 256], 1)

        # apply meanshift to get the new location
        ret, track_window = cv.CamShift(dst, track_window, term_crit)

        # Draw it on image
        img_camshift, pts = hd.draw_camshift(frame, ret)

        # Get camshift extrema
        x, y, w, h = hd.get_camshift_extrema(pts)

        # Create hand window (larger box)
        large_x, large_y, large_w, large_h = hd.get_current_hand_window(x, y, w, h)

        # Draw the larger rectangle on frame
        display_frame = cv.rectangle(img_camshift, (x - int(w / 2), y - int(h / 2)), (x + int(1.5 * w), y + h), (0, 255, 0), 1)

        cv.namedWindow('Display')
        cv.moveWindow('Display', 40, 30)
        cv.imshow("Display", display_frame)

        roi, roi_bgr = hd.get_ROI_image(frame, hsv, (x,y), (large_x, large_y), (large_w, large_h))
        # get silhouette
        mask = hd.get_mask(roi, roi_bgr, roi_hist)
        # get sample
        sample = cf.process_sample(mask)
        # get prediction
        class_name = cf.get_prediction(sample, cnn)
        # display result
        cv.putText(display_frame, class_name, (10, 500), cv.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 255), 2, cv.LINE_AA)
        cv.imshow("Display", display_frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
             break


    cap.release()
    cv.destroyAllWindows()