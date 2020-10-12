from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
from imutils.video import VideoStream
import time
import numpy as np
from ..geometry.rotations import Rotations

class Animation:
    def __init__(self):

        return

    def plot_trajectory_3D(fps, transl, quats, pts3d):
        """
        PLOTTRAJECTORY3D Given the (timestamped) poses from optitrack (time, translations, quaternions), draw the trajectory of the camera (3 colored axes, RGB).

        - fps: framerate of the video
        - transl(N,3): translations
        - quats(N,4): orientations given by quaternions
        - pts3d(N,3): additional 3D points to plot

        transl and quats refer to the transformtion T_W_C that maps points from the camera coordinate frame to the world frame, i.e. the transformation that expresses the camera position in the world frame.
        """

        decimation_factor = 1
        scale_factor_arrow = 0.05
        video_filename = 'motion.avi'
        # use MJPG encoder and *.avi extension

        num_poses = transl.shape[0]

        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        writer = None
        (w, h) = 400, 400 # frame size for video

        # Draw the trajectory
        fig = plt.figure(figsize=(4,4), dpi=100)
        # cap = cv2.VideoCapture(0)
        ax = fig.add_subplot(111, projection='3d')

        for k in range(0, num_poses, decimation_factor):
            pos = transl[k, :] # current position
            rot_mat = Rotations.quat_2_rot_mat(quats[k,:]) # current orientation

            if writer is None:
                # store image dimensions, initialize video writer, construct array of zeros
                writer = cv2.VideoWriter(video_filename, fourcc, fps, (w, h), True)

            if k == 0:
                
                #  Plot translation
                positionHandle, = ax.plot(pos[0], pos[1], pos[2], marker='o')

                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')

                # ax.set_xlim(-0.3, 0.2)
                # ax.set_ylim(-0.1, 0.28)
                # ax.set_zlim(-0.6, 0.)

                # Plot the 3D points
                ax.scatter(pts3d[:,0], pts3d[:,1], pts3d[:,2])

            # Draw only the current moving frame; not the past ones

            # Plot translation
            positionHandle.set_data(pos[0], pos[1])
            positionHandle.set_3d_properties(pos[2])

            # Plot orientation using axes (X=red, Y=green, Z=blue) at current location
            axisX = ax.quiver(pos[0], pos[1], pos[2], rot_mat[0,0], rot_mat[1,0], rot_mat[2,0], length=scale_factor_arrow, color='r', normalize=False)
            axisY = ax.quiver(pos[0], pos[1], pos[2], rot_mat[0,1], rot_mat[1,1], rot_mat[2,1], length=scale_factor_arrow, color='g', normalize=False)
            axisZ = ax.quiver(pos[0], pos[1], pos[2], rot_mat[0,2], rot_mat[1,2], rot_mat[2,2], length=scale_factor_arrow, color='b', normalize=False)

            # Redraw the canvas
            fig.canvas.draw()
            
            # Convert canvas to image
            img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8,
            sep='')
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

            # img is rgb, convert to opencv's default bgr
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            # write the frame to the video file
            writer.write(img)

            # display image with opencv or any operation you like
            # cv2.imshow("plot",img)
            plt.show(block=False)
            plt.pause(0.01)

            axisX.remove()
            axisY.remove()
            axisZ.remove()

        return