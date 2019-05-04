import cv2
import numpy as np

hand_hist = None
traverse_point = []
total_rectangle = 1
x1 = None
y1 = None

width = 50
height = 50

rows,cols = None, None


def rescale_frame(frame, wpercent=130, hpercent=130):
    width = int(frame.shape[1] * wpercent / 100)
    height = int(frame.shape[0] * hpercent / 100)
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def contours(hist_mask_image):
    gray_hist_mask_image = cv2.cvtColor(hist_mask_image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray_hist_mask_image, 0, 255, 0)
    cont, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return cont


def max_contour(contour_list):
    max_i = 0
    max_area = 0

    for i in range(len(contour_list)):
        cnt = contour_list[i]

        area_cnt = cv2.contourArea(cnt)

        if area_cnt > max_area:
            max_area = area_cnt
            max_i = i

        return contour_list[max_i]


def draw_rect(frame):
    global x1 , y1 , width , height, rows, cols

##    width = int(rows*.2)
##    height = int(cols*.2)

    x1 = int(cols*.45)
    y1 = int(rows*.45) 

    cv2.rectangle(frame, (x1, y1),(x1+width, y1+height),(0, 255, 0), 1)
    
    return frame


def hand_histogram(frame):
    global x1 , y1 , width , height, rows, cols

    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    roi = np.zeros([width,height, 3], dtype=hsv_frame.dtype)

    roi[:, :,:] = hsv_frame[x1: x1+width,y1: y1+height,:]

    hand_hist = cv2.calcHist([roi], [0, 1], None, [180, 256], [0, 180, 0, 256])
    return cv2.normalize(hand_hist, hand_hist, 0, 255, cv2.NORM_MINMAX)


def hist_masking(frame, hist):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    dst = cv2.calcBackProject([hsv], [0, 1], hist, [0, 180, 0, 256], 1)

    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    cv2.filter2D(dst, -1, disc, dst)

    ret, thresh = cv2.threshold(dst, 150, 255, cv2.THRESH_BINARY)

    # thresh = cv2.dilate(thresh, None, iterations=5)

    thresh = cv2.merge((thresh, thresh, thresh))

    return cv2.bitwise_and(frame, thresh)


def centroid(max_contour):
    moment = cv2.moments(max_contour)
    if moment['m00'] != 0:
        cx = int(moment['m10'] / moment['m00'])
        cy = int(moment['m01'] / moment['m00'])
        return cx, cy
    else:
        return None


def farthest_point(defects, contour, centroid):
    if defects is not None and centroid is not None:
        s = defects[:, 0][:, 0]
        cx, cy = centroid

        x = np.array(contour[s][:, 0][:, 0], dtype=np.float)
        y = np.array(contour[s][:, 0][:, 1], dtype=np.float)

        xp = cv2.pow(cv2.subtract(x, cx), 2)
        yp = cv2.pow(cv2.subtract(y, cy), 2)
        dist = cv2.sqrt(cv2.add(xp, yp))

        dist_max_i = np.argmax(dist)

        if dist_max_i < len(s):
            farthest_defect = s[dist_max_i]
            farthest_point = tuple(contour[farthest_defect][0])
            return farthest_point
        else:
            return None


def draw_circles(frame, traverse_point):
    if traverse_point is not None:
        for i in range(len(traverse_point)):
            cv2.circle(frame, traverse_point[i], int(5 - (5 * i * 3) / 100), [0, 255, 255], -1)


def manage_image_opr(frame, hand_hist):
    hist_mask_image = hist_masking(frame, hand_hist)

    cv2.imshow("Masked Feed", rescale_frame(hist_mask_image))
    return
    
    contour_list = contours(hist_mask_image)
    max_cont = max_contour(contour_list)

    cnt_centroid = centroid(max_cont)
    cv2.circle(frame, cnt_centroid, 5, [255, 0, 255], -1)

    if max_cont is not None:
        hull = cv2.convexHull(max_cont, returnPoints=False)
        defects = cv2.convexityDefects(max_cont, hull)
        far_point = farthest_point(defects, max_cont, cnt_centroid)
        print("Centroid : " + str(cnt_centroid) + ", farthest Point : " + str(far_point))
        cv2.circle(frame, far_point, 5, [0, 0, 255], -1)
        if len(traverse_point) < 20:
            traverse_point.append(far_point)
        else:
            traverse_point.pop(0)
            traverse_point.append(far_point)

        draw_circles(frame, traverse_point)


def main():
    global hand_hist,rows,cols
    is_hand_hist_created = False
    capture = cv2.VideoCapture(0)

    while capture.isOpened():
        pressed_key = cv2.waitKey(1)
        _, frame = capture.read()
        rows, cols, _ = frame.shape

        if pressed_key & 0xFF == ord('z'):
            is_hand_hist_created = True
            hand_hist = hand_histogram(frame)

        if is_hand_hist_created:
            manage_image_opr(frame, hand_hist)

        else:
            frame = draw_rect(frame)

        cv2.imshow("Live Feed", rescale_frame(frame))

        if pressed_key == 27:
            break

    cv2.destroyAllWindows()
    capture.release()


if __name__ == '__main__':
    main()
