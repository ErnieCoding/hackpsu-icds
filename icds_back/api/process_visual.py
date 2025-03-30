from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
import json
import numpy as np
import open3d as o3d
import os
import uuid

UPLOAD_DIR = os.path.join(settings.MEDIA_ROOT, 'uploads')
os.makedirs(UPLOAD_DIR, exist_ok=True)

@csrf_exempt
def process_point_cloud(request):
    if request.method != "POST":
        return JsonResponse({'status': 'error', 'message': 'Method not allowed'}, status=405)

    try:
        if request.content_type.startswith('multipart/form-data'):
            uploaded_file = request.FILES.get('file')
            if not uploaded_file:
                return JsonResponse({"status": "error", "message": "No file uploaded"}, status=400)

            # Save the file using a unique name
            filename = f"{uuid.uuid4().hex}.pcd"
            save_path = os.path.join(UPLOAD_DIR, filename)
            with open(save_path, 'wb') as out:
                for chunk in uploaded_file.chunks():
                    out.write(chunk)

            file_path = save_path

        else:
            data = json.loads(request.body)
            file_id = data.get("file_id", "")
            if not file_id:
                return JsonResponse({"status": "error", "message": "Missing file_id"}, status=400)

            file_path = os.path.join(UPLOAD_DIR, file_id)

            if not os.path.isfile(file_path):
                return JsonResponse({"status": "error", "message": f"File not found: {file_id}"}, status=404)

        # Process point cloud
        pcd = o3d.io.read_point_cloud(file_path)
        if pcd.is_empty():
            return JsonResponse({"status": "error", "message": "Empty point cloud"}, status=400)

        downpcd = pcd.voxel_down_sample(voxel_size=0.3)
        points = np.asarray(downpcd.points).tolist()
        colors = np.asarray(downpcd.colors).tolist() if downpcd.has_colors() else None
        objects = detect_objects(downpcd)

        return JsonResponse({
            "status": "success",
            "points": points,
            "colors": colors,
            "objects": objects,
            "metadata": {
                "pointCount": len(points)
            },
            "file_id": os.path.basename(file_path)  # return filename only
        })

    except Exception as e:
        return JsonResponse({"status": "error", "message": f"Processing error: {str(e)}"}, status=500)

def detect_objects(pcd):
    points = np.asarray(pcd.points)
    if points.size == 0:
        return []
    return [{
        'id': 1,
        'type': 'object',
        'confidence': 0.9,
        'bounds': {
            'x': {'min': float(np.min(points[:, 0])), 'max': float(np.max(points[:, 0]))},
            'y': {'min': float(np.min(points[:, 1])), 'max': float(np.max(points[:, 1]))},
            'z': {'min': float(np.min(points[:, 2])), 'max': float(np.max(points[:, 2]))},
        },
        'pointCount': len(points)
    }]
