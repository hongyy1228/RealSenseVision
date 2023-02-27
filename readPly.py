import open3d.visualization
from pyntcloud import PyntCloud
from pyntcloud.geometry.models.plane import Plane
import open3d as o3d

myPC = o3d.io.read_point_cloud('out.ply')
o3d.visualization.draw_geometries([myPC])

# Downsample
downpcd = myPC.voxel_down_sample(voxel_size=0.05)
o3d.visualization.draw_geometries([downpcd])

downpcd.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

o3d.visualization.draw_geometries([downpcd], point_show_normal = True)

# Try segmenting plane
plane_model, inliers = myPC.segment_plane(distance_threshold=0.01,
                                         ransac_n=3,
                                         num_iterations=1000)
[a, b, c, d] = plane_model
print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

inlier_cloud = myPC.select_by_index(inliers)
inlier_cloud.paint_uniform_color([1.0, 0, 0])
outlier_cloud = myPC.select_by_index(inliers, invert=True)
o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud],)



myPC.plot()
is_floor = myPC.add_scalar_field('plane_fit')
myPC.plot(use_as_color = is_floor, cmap='cool')