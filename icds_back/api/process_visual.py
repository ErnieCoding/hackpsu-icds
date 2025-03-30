from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET
from django.conf import settings
import json
import numpy as np
import open3d as o3d
import os
import uuid
import sys
import pickle
import re

UPLOAD_DIR = os.path.join(settings.MEDIA_ROOT, 'uploads')
os.makedirs(UPLOAD_DIR, exist_ok=True)

@csrf_exempt
def process_point_cloud(request):
    if request.method != "POST":
        return JsonResponse({'status': 'error', 'message': 'Method not allowed'}, status=405)

    try:
        # 👇 Branch 1: Direct File Upload
        if request.content_type.startswith('multipart/form-data'):
            print("📥 Received file upload")

            uploaded_file = request.FILES.get('file')
            if not uploaded_file:
                return JsonResponse({"status": "error", "message": "No file uploaded"}, status=400)

            filename = f"{uuid.uuid4().hex}.pcd"
            save_path = os.path.join(UPLOAD_DIR, filename)

            print(f"📝 Saving uploaded file to: {save_path}")
            with open(save_path, 'wb') as out:
                for chunk in uploaded_file.chunks():
                    out.write(chunk)

            file_path = save_path
            file_id = filename  # This will be returned

        # 👇 Branch 2: JSON with file_id (existing file on server)
        elif request.content_type == "application/json":
            print("📨 Received JSON file_id request")

            raw_body = request.body.decode('utf-8')
            print("📦 Raw body:", raw_body)

            data = json.loads(raw_body)
            file_id = data.get("file_id", "")
            if not file_id:
                return JsonResponse({"status": "error", "message": "Missing file_id"}, status=400)

            file_path = os.path.join(UPLOAD_DIR, file_id)
            print(f"📂 Looking for file at: {file_path}")

            if not os.path.isfile(file_path):
                return JsonResponse({"status": "error", "message": f"File not found: {file_id}"}, status=404)

        else:
            return JsonResponse({"status": "error", "message": "Unsupported content type"}, status=400)

        print(f"🔍 Loading point cloud from: {file_path}")
        pcd = o3d.io.read_point_cloud(file_path)
        if pcd.is_empty():
            return JsonResponse({"status": "error", "message": "Empty point cloud"}, status=400)

        downpcd = pcd.voxel_down_sample(voxel_size=0.3)
        points = np.asarray(downpcd.points).tolist()
        colors = np.asarray(downpcd.colors).tolist() if downpcd.has_colors() else None
        objects = detect_objects(downpcd)

        print("✅ Processing complete.")
        return JsonResponse({
            "status": "success",
            "points": points,
            "colors": colors,
            "objects": objects,
            "metadata": {
                "pointCount": len(points)
            },
            "file_id": file_id
        })

    except Exception as e:
        print("❌ Exception occurred:", str(e))
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

@require_GET
def view_point_cloud(request):
    file_id = request.GET.get("file_id", "")
    if not file_id:
        return JsonResponse({"status": "error", "message": "Missing file_id"}, status=400)

    file_path = os.path.join(UPLOAD_DIR, file_id)
    if not os.path.isfile(file_path):
        return JsonResponse({"status": "error", "message": f"File not found: {file_id}"}, status=404)

    try:
        print(f"📡 Viewing point cloud from: {file_path}")
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
            "file_id": file_id
        })

    except Exception as e:
        print("❌ Exception in view endpoint:", str(e))
        return JsonResponse({"status": "error", "message": f"View error: {str(e)}"}, status=500)
    
@require_GET
def run_object_detection(request):
    import subprocess

    file_id = request.GET.get("file_id", "")
    if not file_id:
        return JsonResponse({"status": "error", "message": "Missing file_id"}, status=400)

    file_path = os.path.join(UPLOAD_DIR, file_id)
    if not os.path.isfile(file_path):
        return JsonResponse({"status": "error", "message": f"File not found: {file_id}"}, status=404)

    try:
        output_dir = os.path.join(UPLOAD_DIR, f"detection_{file_id}")
        os.makedirs(output_dir, exist_ok=True)

        # Define script paths
        BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        detect_script = os.path.join(BASE_DIR, "icds_detection", "scripts", "cie_detection.py")
        vis_script = os.path.join(BASE_DIR, "icds_detection", "scripts", "visualize_standalone.py")

        if not os.path.exists(detect_script) or not os.path.exists(vis_script):
            return JsonResponse({"status": "error", "message": "Required scripts not found"}, status=500)

        # Run detection
        subprocess.run([
            sys.executable,
            detect_script,
            "--input_dir", os.path.dirname(file_path),
            "--output_dir", output_dir
        ], check=True)

        # Run visualization
        result_path = os.path.join(output_dir, "standalone_detection", "detection_results.txt")
        subprocess.run([
            sys.executable,
            vis_script,
            "--pcd_path", file_path,
            "--result_path", result_path
        ], check=True)

        if not os.path.exists(result_path):
            return JsonResponse({"status": "error", "message": "Detection result file not found"}, status=500)

        # Parse detection results
        with open(result_path, "r") as f:
            content = f.read()

        import re
        pattern = r"Object (\d+):\s+Class: ([^\n]+)\s+Confidence: ([^\n]+)\s+Center: \(([^)]+)\)\s+Dimensions: ([^\n]+)"
        matches = re.findall(pattern, content)

        boxes, labels, scores, class_names, objects = [], [], [], [], []
        class_map = {}

        for match in matches:
            obj_id, class_name, confidence, center_str, dim_str = match
            center = [float(x.strip()) for x in center_str.split(',')]
            dimensions = [float(x.strip()) for x in dim_str.split('x')]
            heading = 0.0

            if class_name not in class_map:
                class_map[class_name] = len(class_map) + 1
                class_names.append(class_name)
            label = class_map[class_name]

            boxes.append(center + dimensions + [heading])
            labels.append(label)
            scores.append(float(confidence))

            x, y, z = center
            dx, dy, dz = dimensions
            objects.append({
                "id": int(obj_id),
                "type": class_name,
                "confidence": float(confidence),
                "pointCount": 0,
                "bounds": {
                    "x": {"min": x - dx / 2, "max": x + dx / 2},
                    "y": {"min": y - dy / 2, "max": y + dy / 2},
                    "z": {"min": z - dz / 2, "max": z + dz / 2}
                }
            })

        # Require modified point cloud to exist
        modified_pcd_path = os.path.join(os.path.dirname(file_path), "modified_output.pcd")
        if not os.path.exists(modified_pcd_path):
            return JsonResponse({
                "status": "error",
                "message": "Modified point cloud not found. Visualization script may have failed to save it."
            }, status=500)

        print(f"✅ Using modified point cloud: {modified_pcd_path}")
        pcd = o3d.io.read_point_cloud(modified_pcd_path)
        down = pcd.voxel_down_sample(0.3)
        points = np.asarray(down.points).tolist()
        colors = np.asarray(down.colors).tolist() if down.has_colors() else None

        return JsonResponse({
            "status": "success",
            "points": points,
            "colors": colors,
            "boxes": boxes,
            "labels": labels,
            "scores": scores,
            "class_names": class_names,
            "objects": objects,
            "metadata": {
                "pointCount": len(points)
            },
            "file_id": file_id
        })

    except subprocess.CalledProcessError as e:
        return JsonResponse({"status": "error", "message": "Script execution failed", "details": str(e)}, status=500)
    except Exception as e:
        return JsonResponse({"status": "error", "message": str(e)}, status=500)





