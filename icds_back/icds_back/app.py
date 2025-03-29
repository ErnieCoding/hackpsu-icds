import open3d as o3d
import numpy as np
import plotly.graph_objects as go
from IPython.display import Javascript, display

# Function to load and combine point clouds
def combine_point_clouds(folder_name, file_count):
    combined = o3d.geometry.PointCloud()
    for i in range(file_count):
        file_path = f"cie_pointcloud/{folder_name}/{folder_name}_{i}.pcd"
        try:
            pcd = o3d.io.read_point_cloud(file_path)
            if not pcd.is_empty():
                combined += pcd
            else:
                print(f"Warning: Empty point cloud loaded from {file_path}")
        except Exception as e:
            print(f"Error loading {file_path}: {str(e)}")
    
    if not combined.is_empty():
        o3d.io.write_point_cloud(f"{folder_name}.pcd", combined, write_ascii=True)
    else:
        print(f"Warning: Combined point cloud for {folder_name} is empty")
    
    return combined

# Combine all the point clouds
ceilingtheater = combine_point_clouds("ceilingtheater", 3)
theater = combine_point_clouds("theater", 7)
main = combine_point_clouds("main", 13)
secondary = combine_point_clouds("secondary", 11)

# Combine all into one point cloud
CIE = main + secondary + theater + ceilingtheater
o3d.io.write_point_cloud("CIE.pcd", CIE, write_ascii=True)

pcd = o3d.io.read_point_cloud('CIE.pcd')
print("Original point cloud:", pcd)
downpcd = pcd.voxel_down_sample(voxel_size=0.3)

try:
    from IPython import get_ipython
    def resize_colab_cell():
        display(Javascript('google.colab.output.setIframeHeight(0, true, {maxHeight: 5000})'))
    
    ip = get_ipython()
    if ip is not None:
        ip.events.register('pre_run_cell', resize_colab_cell)
except:
    pass

def draw(geometries):
    graph_obj = []

    for gm in geometries:
        geometry_type = gm.get_geometry_type()
        pts = np.asarray(gm.points)
        clr = None
        
        if gm.has_colors():
            clr = np.asarray(gm.colors)
        elif gm.has_normals():
            clr = (0.5, 0.5, 0.5) + np.asarray(gm.normals) * 0.5
        else:
            gm.paint_uniform_color((1.0, 0.0, 0.0))
            clr = np.asarray(gm.colors)

        sc = go.Scatter3d(
            x=pts[:,0], 
            y=pts[:,1], 
            z=pts[:,2], 
            mode='markers', 
            marker=dict(size=10, color=clr)
        )
        graph_obj.append(sc)

    fig = go.Figure(
        data=graph_obj,
        layout=dict(
            scene=dict(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False),
                aspectratio=dict(x=1, y=1, z=0.5)
            ),
            autosize=False,
            width=1500,
            height=1000
        )
    )
    
    fig.update_layout(
        scene_camera=dict(
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=0, y=0, z=.75)
        ),
        title="Point Cloud of CIE"
    )
    fig.show()
draw([downpcd])