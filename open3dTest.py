import open3d as o3d
import pyrealsense2 as rs
import numpy as np
import cv2

# Configure depth and color streams
pc = rs.pointcloud()
points = rs.points()

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

profile = pipeline.start(config)
align_to = rs.stream.color
align = rs.align(align_to)

try:
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    profile = aligned_frames.get_profile()
    intrinsics = profile.as_video_stream_profile().get_intrinsics()

    pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(intrinsics.width, intrinsics.height, intrinsics.fx, intrinsics.fy, intrinsics.ppx, intrinsics.ppy)

    depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()
    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())

    img_depth = o3d.geometry.Image(depth_image)
    img_color = o3d.geometry.Image(color_image)

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(img_color, img_depth)

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, pinhole_camera_intrinsic)

    pcd.transform([[1,0,0,0], [0,-1,0,0],[0,0,-1,0],[0,0,0,1]])

    o3d.visualization.draw_geometries([pcd], zoom=0.3412, front=[0,0,0],lookat=[0,0,0],up=[0,0,0])

finally:
    pipeline.stop()



