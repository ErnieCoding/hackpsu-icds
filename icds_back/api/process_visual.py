import os
import uuid
import json
import numpy as np
import open3d as o3d
import tempfile
import subprocess
import pickle
import shutil
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET
from django.conf import settings

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

        # Call OpenPCDet for object detection
        print("🔍 Running object detection with OpenPCDet...")
        objects = run_openpcdet_detection(file_path)
        
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

def run_openpcdet_detection(pcd_path):
    """Run OpenPCDet object detection on the given PCD file"""
    try:
        # Create temporary output directory
        output_dir = tempfile.mkdtemp()
        
        # Path to the OpenPCDet detection script
        script_path = os.path.join(os.path.dirname(__file__), 'run_openpcdet_detection.py')
        
        # Configuration and checkpoint files (adjust these paths as needed)
        cfg_file = os.path.join(settings.BASE_DIR, 'tools/cfgs/cie_models/cie_pointpillar.yaml')
        ckpt = os.path.join(settings.BASE_DIR, 'checkpoints/cie_pointpillar.pth')
        
        # Run the detection script
        cmd = [
            'python', script_path,
            '--data_path', pcd_path,
            '--cfg_file', cfg_file,
            '--ckpt', ckpt,
            '--output_dir', output_dir,
            '--visualize', 'True'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"OpenPCDet detection failed: {result.stderr}")
            return []
        
        # Parse the detection results
        pcd_name = os.path.splitext(os.path.basename(pcd_path))[0]
        result_path = os.path.join(output_dir, f'openpcdet_cie_pointpillar/{pcd_name}.pkl')
        
        if not os.path.exists(result_path):
            print("No detection results found")
            return []
        
        # Load and format the detection results
        with open(result_path, 'rb') as f:
            results = pickle.load(f)
        print(f"{results}")   
        
        # Format results for API response
        objects = []
        for i, box in enumerate(results.get('boxes_lidar', [])):
            x, y, z, dx, dy, dz, heading = box
            class_name = results.get('class_names', ['object'])[i] if i < len(results.get('class_names', [])) else 'object'
            
            objects.append({
                'id': i+1,
                'type': class_name,
                'confidence': float(results.get('scores', [0.9])[i]),
                'bounds': {
                    'x': {'min': float(x - dx/2), 'max': float(x + dx/2)},
                    'y': {'min': float(y - dy/2), 'max': float(y + dy/2)},
                    'z': {'min': float(z - dz/2), 'max': float(z + dz/2)},
                },
                'dimensions': {'x': float(dx), 'y': float(dy), 'z': float(dz)},
                'rotation': float(heading),
                'center': {'x': float(x), 'y': float(y), 'z': float(z)}
            })
        
        # Clean up temporary files
        shutil.rmtree(output_dir)
        
        return objects
    
    except Exception as e:
        print(f"Error running OpenPCDet detection: {str(e)}")
        return []

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
        objects = run_openpcdet_detection(file_path)

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
