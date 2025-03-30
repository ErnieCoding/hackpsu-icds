from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
from process_visual import main, combine_point_clouds
import numpy as np
import open3d as o3d

@csrf_exempt
def process_point_cloud(request):
    if request.method == "POST":
        data = json.loads(request.body)
        is_cie_data = data.get('is_cie_data', False)
        is_pcd = data.get("is_pcd", True)
        file_path = data.get("file_path", "")
    
        try:
            pcd = o3d.io.read_point_cloud(file_path)
            downpcd = pcd.voxel_down_sample(voxel_size=0.3)

            pts = np.asarray(downpcd.points).tolist()

            colors = None
            if downpcd.has_colors():
                colors = np.asarray(downpcd.colors).tolist()

            return JsonResponse({
                'status':'success',
                'points':pts,
                'colors':colors
            })
        except Exception as e:
            return JsonResponse({"status":"error", "message":str(e)}, status=500)
    
    return JsonResponse({'status': 'error', 'message': 'Method not allowed'}, status=405)
