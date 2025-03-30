from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
import numpy as np
import open3d as o3d
import os
import tempfile

@csrf_exempt
def process_point_cloud(request):
    if request.method == "POST":
        try:
            # Handle both JSON data and file uploads
            if request.content_type == 'application/json':
                data = json.loads(request.body)
                is_cie_data = data.get('is_cie_data', False)
                is_pcd = data.get("is_pcd", True)
                file_path = data.get("file_path", "")
                
                # Validate file path
                if not file_path:
                    return JsonResponse({"status": "error", "message": "No file path provided"}, status=400)
                
                if not os.path.exists(file_path):
                    return JsonResponse({"status": "error", "message": f"File not found: {file_path}"}, status=404)
                
            else:  # Handle file upload
                uploaded_file = request.FILES.get('file')
                if not uploaded_file:
                    return JsonResponse({"status": "error", "message": "No file uploaded"}, status=400)
                
                # Create temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pcd') as tmp:
                    for chunk in uploaded_file.chunks():
                        tmp.write(chunk)
                    file_path = tmp.name
            
            # Process the point cloud file
            try:
                pcd = o3d.io.read_point_cloud(file_path)
                if pcd.is_empty():
                    return JsonResponse({"status": "error", "message": "Empty point cloud"}, status=400)
                
                # Down-sample for better performance
                downpcd = pcd.voxel_down_sample(voxel_size=0.3)
                
                # Extract points and colors
                pts = np.asarray(downpcd.points).tolist()
                
                colors = None
                if downpcd.has_colors():
                    colors = np.asarray(downpcd.colors).tolist()
                
                # Perform object detection (mock implementation)
                objects = detect_objects(downpcd)
                
                # Return the processed data
                return JsonResponse({
                    'status': 'success',
                    'points': pts,
                    'colors': colors,
                    'objects': objects,
                    'metadata': {
                        'pointCount': len(pts),
                        
                    }
                })
                
            except Exception as e:
                return JsonResponse({"status": "error", "message": f"Error processing point cloud: {str(e)}"}, status=500)
            
            finally:
                # Clean up temporary file if it was created from an upload
                if request.content_type != 'application/json' and os.path.exists(file_path):
                    os.unlink(file_path)
                
        except Exception as e:
            return JsonResponse({"status": "error", "message": str(e)}, status=500)
    
    return JsonResponse({'status': 'error', 'message': 'Method not allowed'}, status=405)

def detect_objects(pcd):
    """
    Mock function for object detection from point cloud.
    In a real implementation, this would use ML models from process_visual.py
    """
    # Get point cloud bounds
    points = np.asarray(pcd.points)
    min_bound = np.min(points, axis=0)