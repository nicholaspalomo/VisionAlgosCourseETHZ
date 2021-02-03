# Runner script for exercise 6 (Multiple-view Geometry 2, Lecture 8)

from matplotlib import pyplot as plt
import pathlib
import numpy as np
import sys

sys.path.append(".")

np.random.seed(42)

from tools.utils.image import Image
from tools.utils.process_text_file import ProcessTextFile
from tools.algo.eight_point import EightPoint, Triangulation

def run_test_triangulation():

    N = 10 # Number of 3D points
    P = np.random.rand(N,4) # Homogeneous coordinates of 3D points

    # Test linear triangulation
    P[:,2] = P[:,2] * 5 + 10
    P[:,-1] = 1
    
    M1 = np.array([\
        [500., 0., 320., 0.],\
        [0., 500., 240., 0.],\
        [0., 0., 1., 0.]\
        ])

    M2 = np.array([\
        [500., 0., 320., -100.],\
        [0., 500., 240., 0.],\
        [0., 0., 1., 0.]\
        ])

    p1 = np.transpose(np.matmul(M1, P.transpose())) # Image (projected) points
    p2 = np.transpose(np.matmul(M2, P.transpose()))

    P_est = Triangulation.linear_triangulation(p1, p2, M1, M2)

    print('P_est - P = \n{}'.format(P_est - P))

    return

def run_test_8point():

    N = 40 # Number of 3D points
    X = np.random.rand(N,4) # Homogeneous coordinates of 3D points

    # Simulated scene with error-free correspondences
    X[2, :] = X[2, :] * 5 + 10
    X[3, :] = 1

    P1 = np.array([\
        [500., 0., 320., 0.],\
        [0., 500., 240., 0.],\
        [0., 0., 1., 0.]\
        ])

    P2 = np.array([\
        [500., 0., 320., -100.],\
        [0., 500., 240., 0.],\
        [0., 0., 1., 0.]\
        ])

    x1 = np.matmul(X, P1.transpose()) # Image (i.e., projected points)
    x2 = np.matmul(X, P2.transpose())

    sigma = 1e-1
    noisy_x1 = x1 + sigma * np.random.rand(x1.shape[0], x1.shape[1])
    noisy_x2 = x2 + sigma * np.random.rand(x2.shape[0], x2.shape[1])

    ## Fundamental matrix estimation via the 8-point algorithm

    # Estimate fundamental matrix
    # Call the 8-points algorithm on inputs x1, x2
    # Also check the epipolar constraint x2[i].transpose() * F * x1[i] = 0 for all points i.
    F, cost_dist_epi_line, cost_algebraic = EightPoint.fundamental_eight_point(x1, x2, compute_costs=True)

    print("Noise-free correspondences\n")
    print("Algebraic error: {}\n".format(cost_algebraic))
    print("Geometric error: {} px\n".format(cost_dist_epi_line))

    F, cost_dist_epi_line, cost_algebraic = EightPoint.fundamental_eight_point(noisy_x1, noisy_x2, compute_costs=True)

    print("Noise-free correspondences (sigma={}) with fundamentalEightPoint\n".format(sigma))
    print("Algebraic error: {}\n".format(cost_algebraic))
    print("Geometric error: {} px\n".format(cost_dist_epi_line))

    return

def run_sfm():

    imgs_path = str(pathlib.Path(__file__).parent.absolute()) + "/data"

    img = Image()
    img.load_image(imgs_path + '/0001.jpg')
    img2 = img.copy()
    img2.load_image(imgs_path + '/0002.jpg')

    K = np.array([\
        [1379.74, 0., 760.35],\
        [0., 1382.08, 503.41],\
        [0., 0., 1.]\
        ])

    # Load noise-free (i.e. outlier-free) point correspondences

    file = ProcessTextFile()
    p1 = file.read_file(imgs_path + '/matches0001.txt').transpose()
    p2 = file.read_file(imgs_path + '/matches0002.txt').transpose()

    ones = np.ones((p1.shape[0],1))
    p1 = np.concatenate((p1, ones), axis=1)
    p2 = np.concatenate((p2, ones), axis=1)

    # Estimate the essential matrix E using the 8-point algorithm

    eight_point = EightPoint(K, K) # takes as an input the calibration matrices of the two matrices

    E = eight_point.estimate_essential_matrix(p1, p2) # assume that K_1 = K_2 = K

    return

def main():

    run_test_triangulation()

    run_test_8point()

    run_sfm()

    return

if __name__ == '__main__':

    main()