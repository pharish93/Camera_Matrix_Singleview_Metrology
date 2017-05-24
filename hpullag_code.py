#################################################
# Computer Vision Project :-  Calculation of Camera Matrix
# Authors :-
# Harish Pulagurla :- hpullag@ncsu.edu
# Last Edited :- 17th April 2017
#################################################

# import the necessary packages
from __future__ import division
import cv2
import numpy as np
import os
import random
from pylsd import lsd
from sklearn.cluster import KMeans
import math

Image_name = 'ip4.jpg'
Project2 = True
Project1 = False
# Global Constants - parameter tuning
DEBUG = True
Line_Threshold = 25          # Removing smaller lines with length less than Line_Threshold
Line_Threshold_Width = 1     # Removing smaller lines with width less than Threshold

N_iterations = 50           # Number of iterations to run RANSAC
Distance_Threshold = 20     # Distance Threshold for perpendicular distance from point to line

# Creating global variables to be used across functions
mouseX = []
mouseY = []
Vanishing_point = []
raw_dimensions = [0, 0, 0, 0]

max_inliers = 0


# distance between line and point
def line_point_distance(lines, point, all_lines, line_equ, src):
    new_src = src.copy()
    inliers_count = 0
    for i in range(len(lines)):
        normal = math.sqrt(math.pow(lines[i][0], 2) + math.pow(lines[i][1], 2))
        new_line = np.divide(lines[i], normal)      # Normalizing each of the lines
        perpendicular_dist = abs(np.inner(new_line, point)) # Distance of point to each of the Normalized lines
        # print perpendicular_dist
        if perpendicular_dist < Distance_Threshold: # If Perp Dist is less than Threshold, it is an inlier
            inliers_count += 1
            if DEBUG :  # Debugging to visualize inliers in each iteration
                for x in range(all_lines.shape[0]): # iterating through all lines in that cluster
                    if lines[i][0] == line_equ[x][0] and lines[i][1] == line_equ[x][1] and lines[i][2] == line_equ[x][2]:
                        pt1 = (int(all_lines[x, 0]), int(all_lines[x, 1]))
                        pt2 = (int(all_lines[x, 2]), int(all_lines[x, 3]))
                        width = all_lines[x, 4]
                        cv2.line(new_src, pt1, pt2, (255, 0, 0), int(np.ceil(width / 2)))

    if DEBUG:
        #cv2.imshow('Inlier lines', new_src)
        #cv2.waitKey(0)
        pass

    return inliers_count


def intersection_point(line1, line2):
    intersection = np.cross(line1, line2)       # Homogeneous coordinates , cross product gives point of intersection
    if intersection[2] != 0:    # Normalizing the lines
        intersection = np.divide(intersection, intersection[2])
    return intersection

#Function to Automatically Detect Vanishing points base on RANSAC
def auto_detect_vanishing_points(src):
    global Vanishing_point
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

    lines = lsd(gray)  # Line Segment Detector

    # Removing Noisy lines ie lines with length less than 25pixels
    itr = 0
    while itr < (lines.shape[0]):
        length = math.sqrt(
            pow(int(lines[itr, 0]) - int(lines[itr, 2]), 2) + pow(int(lines[itr, 1]) - int(lines[itr, 3]), 2))
        if length < Line_Threshold or lines[itr , 4] < Line_Threshold_Width:
            lines = np.delete(lines, (itr), axis=0)
        itr += 1

    if DEBUG:
        print lines

    line_equ = np.zeros((lines.shape[0], 3))
    sin_val = np.zeros((lines.shape[0], 1))

    if DEBUG:
        print lines.shape[0]

    # Find line equations in homogeneous coordinates and calculate its slope & sine of the angle
    for i in range(lines.shape[0]):
        pt1 = [int(lines[i, 0]), int(lines[i, 1]), 1]
        pt2 = [int(lines[i, 2]), int(lines[i, 3]), 1]
        line_equ[i] = np.cross(pt1, pt2)

        if int(lines[i, 1]) - int(lines[i, 3]) != 0:
            slope = float((int(lines[i, 0]) - int(lines[i, 2]))) / (int(lines[i, 1]) - int(lines[i, 3]))
            sin_val[i] = math.sin(math.atan(slope))
        else:  # when slope is infinity
            sin_val[i] = 1

    # Clustering of slopes into 3 categories using Kmeans
    kmeans = KMeans(n_clusters=3, random_state=0).fit(sin_val)

    # Forming line clusters and storing them into new Matrix variable of M[3][lines]
    Matrix = [[0 for x in range(1)] for y in range(3)]
    src1 = src.copy()
    for i in range(lines.shape[0]):
        pt1 = (int(lines[i, 0]), int(lines[i, 1]))
        pt2 = (int(lines[i, 2]), int(lines[i, 3]))

        width = lines[i, 4]
        colour = (255, 0, 0)
        if kmeans.labels_[i] % 3 == 0:
            colour = (255, 0, 0)
            Matrix[0].append(line_equ[i])
        elif kmeans.labels_[i] % 3 == 1:
            colour = (0, 255, 0)
            Matrix[1].append(line_equ[i])
        elif kmeans.labels_[i] % 3 == 2:
            colour = (0, 0, 255)
            Matrix[2].append(line_equ[i])

        cv2.line(src1, pt1, pt2, colour, int(np.ceil(width / 2)))
    if DEBUG:
        cv2.imshow('lines', src1)
        cv2.waitKey(0)
        # cv2.destroyAllWindows()

    cv2.imwrite('Clustered_Lines_Image.jpg', src1)  # image with 3 cluster - each line put in one of the clusters

    del Matrix[0][0]  # bad coding - first element set to 0 to enable it Matrix.append easily
    del Matrix[1][0]  # bad coding - first element set to 0 to enable it Matrix.append easily
    del Matrix[2][0]  # bad coding - first element set to 0 to enable it Matrix.append easily


    global max_inliers

    for index in range(0, 3):  # iterating along 3 line clusters
        max_inliers = 0
        vp = [0, 0, 0]
        for i in range(0, N_iterations):  # Start Ransac and Repeat N times
            a = random.sample(Matrix[index], 2)  # Selecting 2 lines at random from the lines cluster
            point_vanish = intersection_point(a[0], a[1])   # Determine the possible vanishing point as the intersection
                                                            #  of selected random lines

            if point_vanish[2] != 0:    # Bad Coding - Ignoring the cases where points meet at infinity as
                                        # I am not able handel them while distance calculation
                inliers_count = line_point_distance(Matrix[index], point_vanish, lines, line_equ, src)
            else:
                inliers_count = 0

            if inliers_count > max_inliers:
                max_inliers = inliers_count
                best_lines = a
                vp = point_vanish
                #print max_inliers

        Vanishing_point.append(vp)

        for j in range(2):
            for x in range(line_equ.shape[0]):
                if best_lines[j][0] == line_equ[x][0] and best_lines[j][1] == line_equ[x][1] and best_lines[j][2] == \
                        line_equ[x][2]:
                    pt1 = (int(lines[x, 0]), int(lines[x, 1]))
                    pt2 = (int(lines[x, 2]), int(lines[x, 3]))
                    width = lines[x, 4]
                    color = [0, 0, 0]
                    color[j] = 255
                    color = tuple(color)
                    cv2.line(src, pt1, pt2, color, int(np.ceil(width / 2)))

        if DEBUG:
            cv2.imshow('lines', src)
            cv2.waitKey(0)

    print Vanishing_point


# Reoccuring function to capture mouse double click
def select_points(event, x, y, flags, param):
    global mouseX
    global mouseY
    if event == cv2.EVENT_LBUTTONDBLCLK:
        cv2.circle(param[0], (x, y), 5, param[1], -1)
        mouseX.append(x)
        mouseY.append(y)


# Function to draw lines pased on points selection and obtaining a vanishing point from it
# User is expected  to select 2 - 2 sets of points for each of the dimentions
def draw_lines(img):
    global mouseX
    global mouseY
    global Vanishing_point
    for i in range(0, 3):
        print "Choose 2 sets of points {2 each} for the lines to pass through them"
        color = [0, 0, 0]
        color[i] = 255
        color = tuple(color)
        cv2.namedWindow('image_draw_lines')
        cv2.setMouseCallback('image_draw_lines', select_points, (img, color))
        while (1):
            cv2.imshow('image_draw_lines', img)
            k = cv2.waitKey(20) & 0xFF
            if k == 27 or mouseX.__len__() == 4:
                cv2.imshow('image_draw_lines', img)
                cv2.waitKey(20)
                break

        cv2.line(img, (mouseX[0], mouseY[0]), (mouseX[1], mouseY[1]), color, 5)
        cv2.line(img, (mouseX[2], mouseY[2]), (mouseX[3], mouseY[3]), color, 5)
        cv2.imshow('image_draw_lines', img)
        cv2.waitKey(20)

        # Storing points collected from users double click selection
        p1 = [mouseX[0], mouseY[0], 1]
        p2 = [mouseX[1], mouseY[1], 1]
        p3 = [mouseX[2], mouseY[2], 1]
        p4 = [mouseX[3], mouseY[3], 1]

        # Find line equations of homogenious coordinated by doing cross product of lines
        L1 = np.cross(p1, p2)
        L2 = np.cross(p3, p4)

        # Find points of intersection based on corss product of the line equations
        R_homo = np.cross(L1, L2)
        # convert homogenious coordinates to [x y 1] format
        if R_homo[2] != 0:
            R = np.divide(R_homo, R_homo[2])
        Vanishing_point.append(R)

        # delete the mouse pointes for use in next direction coordinates
        del mouseX[:]
        del mouseY[:]


# Function to draw the axis of the system
# and  enter the coorsponding distance from the origin which is to be used for scaling
def draw_axis(img):
    global mouseX
    global mouseY
    print "click to select each of the origin & axis end points and the distance in the physical world you know"
    for i in range(0, 4):
        color = [0, 0, 0]
        color[(i - 1) % 3] = 255
        color = tuple(color)
        cv2.namedWindow('image_draw_axis')
        cv2.setMouseCallback('image_draw_axis', select_points, color)
        while (1):
            cv2.imshow('image_draw_axis', img)
            k = cv2.waitKey(20) & 0xFF
            if k == 27 or mouseX.__len__() == (i + 1):
                cv2.circle(img, (mouseX[i], mouseY[i]), 5, color, -1)
                if i != 0:
                    cv2.line(img, (mouseX[0], mouseY[0]), (mouseX[i], mouseY[i]), color, 5)
                cv2.imshow('image_draw_axis', img)
                cv2.waitKey(20)
                break
        if i != 0:
            raw_dimensions[i] = input("enter distance from origin point")


#######################################################################################################################
# Main program starts here
#######################################################################################################################
def main():
    global mouseX
    global mouseY
    global Vanishing_point
    # modify the file name to change the input image
    img_original = cv2.imread(Image_name)
    img = img_original.copy()
    if Project2:
        auto_detect_vanishing_points(img)
    elif Project1:
        draw_lines(img)


    img = img_original.copy()
    draw_axis(img)

    # origin and reference point selected
    # converting each of them into ndarray column format
    Origin = [mouseX[0], mouseY[0], 1]
    O = np.array([Origin]).T

    R1 = np.array([[mouseX[1], mouseY[1], 1]]).T
    R2 = np.array([[mouseX[2], mouseY[2], 1]]).T
    R3 = np.array([[mouseX[3], mouseY[3], 1]]).T

    print "Vanishing points are :", Vanishing_point

    # Spliting and saving into seperate variables in ndarray column format
    Vx = np.array([list(Vanishing_point[0])]).T
    Vy = np.array([list(Vanishing_point[1])]).T
    Vz = np.array([list(Vanishing_point[2])]).T

    # Computing the scaling factor by an inverse matrix calculation
    a_x = (np.linalg.lstsq(Vx, R1 - O)[0]) / raw_dimensions[1]
    b_y = (np.linalg.lstsq(Vy, R2 - O)[0]) / raw_dimensions[2]
    c_z = (np.linalg.lstsq(Vz, R3 - O)[0]) / raw_dimensions[3]

    Vx_new = (Vx * a_x[0][0]).flatten()
    Vy_new = (Vy * b_y[0][0]).flatten()
    Vz_new = (Vz * c_z[0][0]).flatten()

    # Projective matrix as an accumulation of vanishing points in each direction and origin as the final column
    P = np.array([Vx_new, Vy_new, Vz_new, Origin]).T

    print "projection matrix \n", P
    print

    img = img_original.copy()

    # Computing Texture maps for each of the planes
    # xy image plane
    Hxy = P[:, [0, 1, 3]]
    img_xy = cv2.warpPerspective(img, Hxy, (3000, 3000), flags=cv2.WARP_INVERSE_MAP)
    cv2.imshow('img_xy', img_xy)
    cv2.imwrite('img_xy_plane.png', img_xy)

    # xz image plane
    Hxz = P[:, [0, 2, 3]]
    img_xz = cv2.warpPerspective(img, Hxz, (1000, 1000), flags=cv2.WARP_INVERSE_MAP)
    cv2.imshow('img_xz', img_xz)
    cv2.imwrite('img_xz_plane.png', img_xz)

    # yz image plane
    Hyz = P[:, [1, 2, 3]]
    img_yz = cv2.warpPerspective(img, Hyz, (1000, 1000), flags=cv2.WARP_INVERSE_MAP)
    cv2.imshow('img_yz', img_yz)
    cv2.imwrite('img_yz_plane.png', img_yz)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
