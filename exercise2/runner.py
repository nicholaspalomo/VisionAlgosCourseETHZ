import cv2 # OpenCV
import matplotlib as mtplb
import numpy as np
import argparse as ap
import pathlib
from matplotlib import pyplot as plt
import os
import sys
sys.path.append(".")

from tools.utils.process_text_file import ProcessTextFile
from tools.algo.perspective_projection import PerspectiveProjection
from tools.algo.dlt import DLT

def main():
    current_file_path = str(pathlib.Path(__file__).parent.absolute()) + "/data"

    text_file_handler = ProcessTextFile()
    camera_K_matrix = text_file_handler.read_file(current_file_path + "/K.txt")
    p_W_corners = text_file_handler.read_file(current_file_path + "/p_W_corners.txt")
    detected_corners = text_file_handler.read_file(current_file_path + "/detected_corners.txt")

    dlt = DLT(camera_K_matrix, np.array([0., 0.]))

    image_coordinates = dlt.reproject_points(detected_corners, p_W_corners)

    plt.figure()
    dlt.draw_line_in_I_from_points_in_I(
        current_file_path + "/images_undistorted/img_0001.jpg", 
        image_coordinates, 
        linestyle='o')
    plt.pause(10)
    plt.close()

if __name__ == "__main__":
    main()