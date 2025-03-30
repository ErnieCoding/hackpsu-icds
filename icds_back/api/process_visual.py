import open3d as o3d
import numpy as np
import plotly.graph_objects as go

def pcd_to_llm_text(pcd_path, max_points=1000):
    """Convert PCD to LLM-readable text using Open3D"""
    pcd = o3d.io.read_point_cloud(pcd_path)
    points = np.asarray(pcd.points)
    if len(points) > max_points:
        points = points[np.random.choice(len(points), max_points, replace=False)]
    return "\n".join([f"{x:.2f},{y:.2f},{z:.2f}" for x, y, z in points])

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

def draw_point_cloud(pcd):
    pts = np.asarray(pcd.points)
    clr = None
    
    if pcd.has_colors():
        clr = np.asarray(pcd.colors)
    elif pcd.has_normals():
        clr = (0.5, 0.5, 0.5) + np.asarray(pcd.normals) * 0.5
    else:
        pcd.paint_uniform_color((1.0, 0.0, 0.0))
        clr = np.asarray(pcd.colors)

    fig = go.Figure(
        data=[go.Scatter3d(
            x=pts[:,0], 
            y=pts[:,1], 
            z=pts[:,2], 
            mode='markers', 
            marker=dict(size=10, color=clr)
        )],
        layout=dict(
            scene=dict(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False),
                aspectratio=dict(x=1, y=1, z=0.5)
            ),
            autosize=False,
            width=1500,
            height=1000,
            title="Point Cloud Visualization"
        )
    )
    
    fig.update_layout(
        scene_camera=dict(
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=0, y=0, z=.75)
        )
    )
    fig.show()

def display_image(img):
    # Convert Open3D image to numpy array
    img_np = np.asarray(img)
    
    # Create figure
    fig = go.Figure()
    
    # Add image trace
    fig.add_trace(go.Image(z=img_np))
    
    # Update layout
    fig.update_layout(
        title="Image Visualization",
        autosize=False,
        width=1000,
        height=800,
    )
    
    fig.show()

def main(CIE_data, pcd_or_img, data):
    if CIE_data:
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
        draw_point_cloud(downpcd)
    
    else:
        if pcd_or_img:
            if data.lower().endswith(('.pcd')):
                pcd = o3d.io.read_point_cloud(data)
                print("Original point cloud:", pcd)
                downpcd = pcd.voxel_down_sample(voxel_size=0.3)
                draw_point_cloud(downpcd)
            else:
                print("Error: Unsupported point cloud format. Please provide a .pcd file")
        else:
            if data.lower().endswith(('.png', '.jpg', '.jpeg')):
                img = o3d.io.read_image(data)
                print("Original image dimensions:", np.asarray(img).shape)
                display_image(img)
            else:
                print("Error: Unsupported image format. Please provide .png, .jpg, or .jpeg")
 
# Example usage:
# For point clouds:
# main(True, True, "CIE.pcd")  # For CIE data
# main(False, True, "your_point_cloud.pcd")  # For other point clouds
# all data folder or images should be in the hackpsu-icds directory
# For images:
# main(False, False, "your_image.jpg")
# main(True, True, "CIE.pcd")
"""
TODO: run through vision model to figure out what items there are


"""