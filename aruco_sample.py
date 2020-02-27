#!/usr/bin/env python3

import cv2
import pickle
import numpy as np
from cv2 import aruco
from collections import deque

class Arucos:
    def __init__(self):
        # self.camera_type = 'C922'
        self.camera_type = 'C930E'

        self.load_calibrate()

        # video capture
        self.cap   = cv2.VideoCapture("/dev/{}".format(self.camera_type))
        self.cap.set(3, self.img_size[0]) # 1024 (frame width)
        self.cap.set(4, self.img_size[1]) # 576  (frame height)

        # update camera matrix
        self.update_matrix()

        # aruco dictionary
        self.aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
        self.parameters = aruco.DetectorParameters_create()

        # parameter
        self.show_angle = False
        self.buffer_pts = 64
        self.pts        = deque(maxlen=self.buffer_pts)

    def load_calibrate(self):
        self.camera_dist  = "data/{}_dist.p".format(self.camera_type)

        with open(self.camera_dist, "rb") as f:
            dist_pickle = pickle.load(f)
            print('Successful load: {}'.format(self.camera_dist))

        self.mtx      = dist_pickle["mtx"]
        self.dist     = dist_pickle["dist"]
        self.img_size = dist_pickle["img_size"]

    def update_matrix(self):
        _, img   = self.cap.read()
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        h,  w    = img_gray.shape[:2]

        print('Image size: {}, {}'.format(w, h) )
        self.new_mtx, self.roi = cv2.getOptimalNewCameraMatrix(self.mtx, self.dist, (w,h), 1, (w,h))

    def undistort_image(self, img):
        undist_img =  cv2.undistort(img, self.mtx, self.dist, None, self.new_mtx)
            
        # crop undistorted field
        x, y, w, h = self.roi
        return undist_img[y:y+h, x:x+w]
    
    # Checks if a matrix is a valid rotation matrix.
    def is_rotation_matrix(self, R) :
        Rt = np.transpose(R)
        shouldBeIdentity = np.dot(Rt, R)
        I = np.identity(3, dtype = R.dtype)
        n = np.linalg.norm(I - shouldBeIdentity)
        return n < 1e-6
    
    # Calculates rotation matrix to euler angles
    def rotation_matrix_to_euler_angles(self, R) :
        assert(self.is_rotation_matrix(R))     
        sy = np.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])     
        singular = sy < 1e-6 
        if  not singular :
            x = np.arctan2(R[2,1] , R[2,2])
            y = np.arctan2(-R[2,0], sy)
            z = np.arctan2(R[1,0], R[0,0])
        else :
            x = np.arctan2(-R[1,2], R[1,1])
            y = np.arctan2(-R[2,0], sy)
            z = 0
    
        return np.array([x, y, z])

    def run(self):
        orange = (0, 144, 255)
        red    = (0, 0, 255)

        while True:
            _, img   = self.cap.read()
            # frame    = img.copy()
            frame    = self.undistort_image(img)
            gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, rejectedImgPoints = aruco.detectMarkers(gray_img, self.aruco_dict, parameters=self.parameters)
            
            if np.all(ids != None):
                for i in range(0, ids.size):
                    M = cv2.moments(corners[i])

                    self.cx = int(M["m10"] / M["m00"])
                    self.cy = int(M["m01"] / M["m00"])
                    center = (self.cx, self.cy)
                    self.pts.appendleft(center)

                    rvec, tvec ,_ = aruco.estimatePoseSingleMarkers(corners[i], 0.05, self.mtx, self.dist)
                    aruco.drawAxis(frame, self.mtx, self.dist, rvec, tvec, 0.1)

                    # draw center circle
                    cv2.circle(frame, center, 5, orange, -1)

                    # rotation matrix
                    if self.show_angle:
                        rmat, _    = cv2.Rodrigues(rvec)
                        rx, ry, rz = np.degrees(self.rotation_matrix_to_euler_angles(rmat))
                        cv2.putText(frame, "angle= {:.2f}".format(rz), (self.cx, self.cy+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, red, 1)

                    # coordinate text
                    cv2.putText(frame, "x:{:.0f}".format(self.cx), (self.cx+40, self.cy),    cv2.FONT_HERSHEY_SIMPLEX, 0.5, red, 1)
                    cv2.putText(frame, "y:{:.0f}".format(self.cy), (self.cx+40, self.cy+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, red, 1)

                    # loop over the set of tracked points
                    for i in np.arange(1, len(self.pts)):
                        # if either of the tracked points are None, ignore
                        if self.pts[i - 1] is None or self.pts[i] is None:
                            continue
                        
                        thickness = int(np.sqrt(self.buffer_pts / float(i + 1)) * 2.5)
                        cv2.line(frame, self.pts[i - 1], self.pts[i], orange, thickness)

                aruco.drawDetectedMarkers(frame, corners, ids)
            
            # show frame
            cv2.imshow("aruco", frame)

            # wait key
            k = cv2.waitKey(1)
            if k == 27 or k == ord('q'):
                print('Exit with code 0')
                break

        cv2.destroyAllWindows()

if __name__ == '__main__':
    arucos = Arucos()
    arucos.run()        