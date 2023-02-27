import pyrealsense2 as rs
import numpy as np
import math

pipeline = rs.pipeline()
config = rs.config()
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

config.enable_stream(rs.stream.depth, rs.format.z16, 30)
config.enable_stream(rs.stream.color, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

# Get stream profile and camera intrinsics
profile = pipeline.get_active_profile()
depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
depth_intrinsics = depth_profile.get_intrinsics()
w, h = depth_intrinsics.width, depth_intrinsics.height

# Processing blocks
pc = rs.pointcloud()
colorizer = rs.colorizer()

for x in range(5):
  pipeline.wait_for_frames()

frameset = pipeline.wait_for_frames()
color_frame = frameset.get_color_frame()
depth_frame = frameset.get_depth_frame()


depth_intrinsics = rs.video_stream_profile(
            depth_frame.profile).get_intrinsics()
w, h = depth_intrinsics.width, depth_intrinsics.height

depth_image_np = np.asanyarray(depth_frame.get_data())
color_image_np = np.asanyarray(color_frame.get_data())

pipeline.stop()
print("Frames Captured")

center_pt = (424, 240)
dist_to_center = depth_frame.get_distance(center_pt[0],center_pt[1])
dist_to_center = round(dist_to_center, 4)
depth_point = rs.rs2_deproject_pixel_to_point(depth_intrinsics, center_pt, dist_to_center)

z_degree = 60
actual_length = depth_point[2] * math.sin(np.deg2rad(z_degree))
height_offset = math.sqrt((depth_point[2] ** 2) - (actual_length ** 2))
depth_point = [i * 100 for i in depth_point]
depth_point = [round(i, 5) for i in depth_point]






points = pc.calculate(depth_frame)
vertices = points.get_vertices()
tex_coords = points.get_texture_coordinates()
center_pt = (424, 240)
current_color = rs.texture_coordinate()




depth_intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
center_pt = (424, 240)
center_xyz = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [center_pt[1], center_pt[0]],
                                                 depth_image_np[center_pt[1], center_pt[0]])
center_xyz = np.array(center_xyz)
