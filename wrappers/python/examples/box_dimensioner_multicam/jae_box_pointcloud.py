###########################################################################################################################
##                      License: Apache 2.0. See LICENSE file in root directory.                                         ##
###########################################################################################################################
##                  Simple Box Dimensioner with multiple cameras: Main demo file                                         ##
###########################################################################################################################
## Workflow description:                                                                                                 ##
## 1. Place the calibration chessboard object into the field of view of all the realsense cameras.                       ##
##    Update the chessboard parameters in the script in case a different size is chosen.                                 ##
## 2. Start the program.                                                                                                 ##
## 3. Allow calibration to occur and place the desired object ON the calibration object when the program asks for it.    ##
##    Make sure that the object to be measured is not bigger than the calibration object in length and width.            ##
## 4. The length, width and height of the bounding box of the object is then displayed in millimeters.                   ##
###########################################################################################################################

# Import RealSense, OpenCV and NumPy
import pyrealsense2 as rs
import cv2
import numpy as np
import time
import math

# Import helper functions and classes written to wrap the RealSense, OpenCV and Kabsch Calibration usage
from collections import defaultdict
from realsense_device_manager import DeviceManager
from calibration_kabsch import PoseEstimation
from helper_functions import get_boundary_corners_2D
from measurement_task import calculate_boundingbox_points, calculate_cumulative_pointcloud, visualise_measurements

class AppState:

    def __init__(self, *args, **kwargs):
        self.WIN_NAME = 'RealSense'
        self.pitch, self.yaw = math.radians(-10), math.radians(-15)
        self.translation = np.array([0, 0, -1], dtype=np.float32)
        self.distance = 2
        self.prev_mouse = 0, 0
        self.mouse_btns = [False, False, False]
        self.paused = False
        self.decimate = 1
        self.scale = True
        self.color = True

    def reset(self):
        self.pitch, self.yaw, self.distance = 0, 0, 2
        self.translation[:] = 0, 0, -1

    @property
    def rotation(self):
        Rx, _ = cv2.Rodrigues((self.pitch, 0, 0))
        Ry, _ = cv2.Rodrigues((0, self.yaw, 0))
        return np.dot(Ry, Rx).astype(np.float32)

    @property
    def pivot(self):
        return self.translation + np.array((0, 0, self.distance), dtype=np.float32)

def mouse_cb(event, x, y, flags, param):

    if event == cv2.EVENT_LBUTTONDOWN:
        state.mouse_btns[0] = True

    if event == cv2.EVENT_LBUTTONUP:
        state.mouse_btns[0] = False

    if event == cv2.EVENT_RBUTTONDOWN:
        state.mouse_btns[1] = True

    if event == cv2.EVENT_RBUTTONUP:
        state.mouse_btns[1] = False

    if event == cv2.EVENT_MBUTTONDOWN:
        state.mouse_btns[2] = True

    if event == cv2.EVENT_MBUTTONUP:
        state.mouse_btns[2] = False

    if event == cv2.EVENT_MOUSEMOVE:

        h, w = out.shape[:2]
        dx, dy = x - state.prev_mouse[0], y - state.prev_mouse[1]

        if state.mouse_btns[0]:
            state.yaw += float(dx) / w * 2
            state.pitch -= float(dy) / h * 2

        elif state.mouse_btns[1]:
            dp = np.array((dx / w, dy / h, 0), dtype=np.float32)
            state.translation -= np.dot(state.rotation, dp)

        elif state.mouse_btns[2]:
            dz = math.sqrt(dx**2 + dy**2) * math.copysign(0.01, -dy)
            state.translation[2] += dz
            state.distance -= dz

    if event == cv2.EVENT_MOUSEWHEEL:
        dz = math.copysign(0.1, flags)
        state.translation[2] += dz
        state.distance -= dz

    state.prev_mouse = (x, y)



def project(out,v):
    """project 3d vector array to 2d"""
    h, w = out.shape[:2]
    view_aspect = float(h)/w

    # ignore divide by zero for invalid depth
    with np.errstate(divide='ignore', invalid='ignore'):
        proj = v[:, :-1] / v[:, -1, np.newaxis] * \
            (w*view_aspect, h) + (w/2.0, h/2.0)

    # near clipping
    znear = 0.03
    proj[v[:, 2] < znear] = np.nan
    return proj


def view(v):
    """apply view transformation on vector array"""
    return np.dot(v - state.pivot, state.rotation) + state.pivot - state.translation


def line3d(out, pt1, pt2, color=(0x80, 0x80, 0x80), thickness=1):
    """draw a 3d line from pt1 to pt2"""
    p0 = project(out,pt1.reshape(-1, 3))[0]
    p1 = project(out,pt2.reshape(-1, 3))[0]
    if np.isnan(p0).any() or np.isnan(p1).any():
        return
    p0 = tuple(p0.astype(int))
    p1 = tuple(p1.astype(int))
    rect = (0, 0, out.shape[1], out.shape[0])
    inside, p0, p1 = cv2.clipLine(rect, p0, p1)
    if inside:
        cv2.line(out, p0, p1, color, thickness, cv2.LINE_AA)


def grid(out, pos, rotation=np.eye(3), size=1, n=10, color=(0x80, 0x80, 0x80)):
    """draw a grid on xz plane"""
    pos = np.array(pos)
    s = size / float(n)
    s2 = 0.5 * size
    for i in range(0, n+1):
        x = -s2 + i*s
        line3d(out, view(pos + np.dot((x, 0, -s2), rotation)),
               view(pos + np.dot((x, 0, s2), rotation)), color)
    for i in range(0, n+1):
        z = -s2 + i*s
        line3d(out, view(pos + np.dot((-s2, 0, z), rotation)),
               view(pos + np.dot((s2, 0, z), rotation)), color)


def axes(out, pos, rotation=np.eye(3), size=0.075, thickness=2):
    """draw 3d axes"""
    line3d(out, pos, pos +
           np.dot((0, 0, size), rotation), (0xff, 0, 0), thickness)
    line3d(out, pos, pos +
           np.dot((0, size, 0), rotation), (0, 0xff, 0), thickness)
    line3d(out, pos, pos +
           np.dot((size, 0, 0), rotation), (0, 0, 0xff), thickness)


def frustum(out, intrinsics, color=(0x40, 0x40, 0x40)):
    """draw camera's frustum"""
    orig = view([0, 0, 0])
    w, h = intrinsics.width, intrinsics.height

    for d in range(1, 6, 2):
        def get_point(x, y):
            p = rs.rs2_deproject_pixel_to_point(intrinsics, [x, y], d)
            line3d(out, orig, view(p), color)
            return p

        top_left = get_point(0, 0)
        top_right = get_point(w, 0)
        bottom_right = get_point(w, h)
        bottom_left = get_point(0, h)

        line3d(out, view(top_left), view(top_right), color)
        line3d(out, view(top_right), view(bottom_right), color)
        line3d(out, view(bottom_right), view(bottom_left), color)
        line3d(out, view(bottom_left), view(top_left), color)


def pointcloud(out, verts, texcoords, color, painter=True):
    """draw point cloud with optional painter's algorithm"""
    if painter:
        # Painter's algo, sort points from back to front

        # get reverse sorted indices by z (in view-space)
        # https://gist.github.com/stevenvo/e3dad127598842459b68
        v = view(verts)
        s = v[:, 2].argsort()[::-1]
        proj = project(out,v[s])
    else:
        proj = project(out,view(verts))

    if state.scale:
        proj *= 0.5**state.decimate

    h, w = out.shape[:2]

    # proj now contains 2d image coordinates
    j, i = proj.astype(np.uint32).T

    # create a mask to ignore out-of-bound indices
    im = (i >= 0) & (i < h)
    jm = (j >= 0) & (j < w)
    m = im & jm

    cw, ch = color.shape[:2][::-1]
    if painter:
        # sort texcoord with same indices as above
        # texcoords are [0..1] and relative to top-left pixel corner,
        # multiply by size and add 0.5 to center
        v, u = (texcoords[s] * (cw, ch) + 0.5).astype(np.uint32).T
    else:
        v, u = (texcoords * (cw, ch) + 0.5).astype(np.uint32).T
    # clip texcoords to image
    np.clip(u, 0, ch-1, out=u)
    np.clip(v, 0, cw-1, out=v)

    # perform uv-mapping
    out[i[m], j[m]] = color[u[m], v[m]]



def run_demo():
	
	# Define some constants 
	resolution_width = w = 640 # pixels
	resolution_height = h = 480 # pixels
	frame_rate = 30  # fps
	dispose_frames_for_stablisation = 30  # frames
	
	chessboard_width = 6 # squares
	chessboard_height = 9 	# squares
	square_size = 0.055 # 0.0253 # meters  
	print("Jae")
	try:
		# Enable the streams from all the intel realsense devices
		rs_config = rs.config()
		rs_config.enable_stream(rs.stream.depth, resolution_width, resolution_height, rs.format.z16, frame_rate)
		rs_config.enable_stream(rs.stream.infrared, 1, resolution_width, resolution_height, rs.format.y8, frame_rate)
		rs_config.enable_stream(rs.stream.color, resolution_width, resolution_height, rs.format.bgr8, frame_rate)

		# Use the device manager class to enable the devices and get the frames
		device_manager = DeviceManager(rs.context(), rs_config)
		device_manager.enable_all_devices()
		
		# Allow some frames for the auto-exposure controller to stablise
		for frame in range(dispose_frames_for_stablisation):
			frames = device_manager.poll_frames()

		assert( len(device_manager._available_devices) > 0 )
		"""
		1: Calibration
		Calibrate all the available devices to the world co-ordinates.
		For this purpose, a chessboard printout for use with opencv based calibration process is needed.
		
		"""
		# Get the intrinsics of the realsense device 
		intrinsics_devices = device_manager.get_device_intrinsics(frames)
		
				# Set the chessboard parameters for calibration 
		chessboard_params = [chessboard_height, chessboard_width, square_size] 
		
		# Estimate the pose of the chessboard in the world coordinate using the Kabsch Method
		calibrated_device_count = 0
		while calibrated_device_count < len(device_manager._available_devices):
			frames = device_manager.poll_frames()
			pose_estimator = PoseEstimation(frames, intrinsics_devices, chessboard_params)
			transformation_result_kabsch  = pose_estimator.perform_pose_estimation()
			object_point = pose_estimator.get_chessboard_corners_in3d()
			calibrated_device_count = 0
			for device in device_manager._available_devices:
				if not transformation_result_kabsch[device][0]:
					print("Place the chessboard on the plane where the object needs to be detected..")
				else:
					calibrated_device_count += 1

		# Save the transformation object for all devices in an array to use for measurements
		transformation_devices={}
		chessboard_points_cumulative_3d = np.array([-1,-1,-1]).transpose()
		for device in device_manager._available_devices:
			transformation_devices[device] = transformation_result_kabsch[device][1].inverse()
			points3D = object_point[device][2][:,object_point[device][3]]
			points3D = transformation_devices[device].apply_transformation(points3D)
			chessboard_points_cumulative_3d = np.column_stack( (chessboard_points_cumulative_3d,points3D) )

		# Extract the bounds between which the object's dimensions are needed
		# It is necessary for this demo that the object's length and breath is smaller than that of the chessboard
		chessboard_points_cumulative_3d = np.delete(chessboard_points_cumulative_3d, 0, 1)
		roi_2D = get_boundary_corners_2D(chessboard_points_cumulative_3d)

		print("Calibration completed... \nPlace the box in the field of view of the devices...")


		"""
				2: Measurement and display
				Measure the dimension of the object using depth maps from multiple RealSense devices
				The information from Phase 1 will be used here

				"""

		# Enable the emitter of the devices
		device_manager.enable_emitter(True)

		# Load the JSON settings file in order to enable High Accuracy preset for the realsense
		device_manager.load_settings_json("./HighResHighAccuracyPreset.json")

		# Get the extrinsics of the device to be used later
		extrinsics_devices = device_manager.get_depth_to_color_extrinsics(frames)

		# Get the calibration info as a dictionary to help with display of the measurements onto the color image instead of infra red image
		calibration_info_devices = defaultdict(list)
		for calibration_info in (transformation_devices, intrinsics_devices, extrinsics_devices):
			for key, value in calibration_info.items():
				calibration_info_devices[key].append(value)

		# Set intial state for pointcloud viewing
		#state = AppState()
		# Processing blocks
		pc = rs.pointcloud()
		decimate = rs.decimation_filter()
		decimate.set_option(rs.option.filter_magnitude, 2 ** state.decimate)
		colorizer = rs.colorizer()

		cv2.namedWindow(state.WIN_NAME, cv2.WINDOW_AUTOSIZE)
		cv2.resizeWindow(state.WIN_NAME, w, h)
		cv2.setMouseCallback(state.WIN_NAME, mouse_cb)


		out = np.empty((resolution_height, resolution_width, 3), dtype=np.uint8)

		# Continue acquisition until terminated with Ctrl+C by the user
		while 1:
			 # Get the frames from all the devices
			frames_devices = device_manager.poll_frames()

			# Use a threshold of 5 centimeters from the chessboard as the area where useful points are found
			point_cloud_cumulative = np.array([-1, -1, -1]).transpose()
			for (device, frame) in frames_devices.items() :
				depth_frame = frame[rs.stream.depth]
				depth_frame = decimate.process(depth_frame)

				color_frame = frame[rs.stream.depth]

				depth_image = np.asanyarray(depth_frame.get_data())
				color_image = np.asanyarray(color_frame.get_data())

				depth_colormap = np.asanyarray(colorizer.colorize(depth_frame).get_data())

				mapped_frame, color_source = color_frame, color_image

				points = pc.calculate(depth_frame)
				pc.map_to(mapped_frame)
			# accumulate pointclould
			# Pointcloud data to arrays
			v, t = points.get_vertices(), points.get_texture_coordinates()
			verts = np.asanyarray(v).view(np.float32).reshape(-1, 3)  # xyz
			texcoords = np.asanyarray(t).view(np.float32).reshape(-1, 2)  # uv


			# Render
			now = time.time()

			out.fill(0)

			grid(out, (0, 0.5, 1), size=1, n=10)
			cam = device
			depth_intrinsics = calibration_info_devices[cam][1][rs.stream.depth]
			frustum(out, depth_intrinsics)
			axes(out, view([0, 0, 0]), state.rotation, size=0.1, thickness=1)

			if not state.scale or out.shape[:2] == (h, w):
				pointcloud(out, verts, texcoords, color_source)
			else:
				tmp = np.zeros((h, w, 3), dtype=np.uint8)
				pointcloud(tmp, verts, texcoords, color_source)
				tmp = cv2.resize(
					tmp, out.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
				np.putmask(out, tmp > 0, tmp)

			if any(state.mouse_btns):
				axes(out, view(state.pivot), state.rotation, thickness=4)

			dt = time.time() - now

			cv2.setWindowTitle(
				state.WIN_NAME, "RealSense (%dx%d) %dFPS (%.2fms) %s" %
				(w, h, 1.0/dt, dt*1000, "PAUSED" if state.paused else ""))

			cv2.imshow(state.WIN_NAME, out)
			key = cv2.waitKey(1)

			if key == ord("r"):
				state.reset()

			if key == ord("p"):
				state.paused ^= True

			if key == ord("d"):
				state.decimate = (state.decimate + 1) % 3
				decimate.set_option(rs.option.filter_magnitude, 2 ** state.decimate)

			if key == ord("z"):
				state.scale ^= True

			if key == ord("c"):
				state.color ^= True

			if key == ord("s"):
				cv2.imwrite('./out.png', out)

			if key == ord("e"):
				points.export_to_ply('./out.ply', mapped_frame)

			if key in (27, ord("q")) or cv2.getWindowProperty(state.WIN_NAME, cv2.WND_PROP_AUTOSIZE) < 0:
				break


				# Calculate the pointcloud using the depth frames from all the devices
				#point_cloud = calculate_cumulative_pointcloud(frames_devices, calibration_info_devices, roi_2D)
				#visualize_cumulative_pointcloud(frames_devices, calibration_info_devices, roi_2D)
				# Get the bounding box for the pointcloud in image coordinates of the color imager
				#bounding_box_points_color_image, length, width, height = calculate_boundingbox_points(point_cloud, calibration_info_devices )

				# Draw the bounding box points on the color image and visualise the results
				#visualise_measurements(frames_devices, bounding_box_points_color_image, length, width, height)

	except KeyboardInterrupt:
		print("The program was interupted by the user. Closing the program...")
	
	finally:
		device_manager.disable_streams()
		cv2.destroyAllWindows()
	
state = AppState()

if __name__ == "__main__":
	run_demo()
