import os
from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename
from models import db, SatelliteImage, DetectionTask, Landslide
from services.inference import inference_service
import threading
from datetime import datetime

inference_bp = Blueprint('inference', __name__)

ALLOWED_EXTENSIONS = {'h5', 'tif', 'tiff'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def run_inference_task(task_id, file_path):
    """
    Background thread function to run inference and update DB.
    """
    # Need to push app context to use db inside thread
    from app import create_app
    app = create_app(os.getenv('FLASK_CONFIG') or 'default')
    
    with app.app_context():
        task = DetectionTask.query.get(task_id)
        if not task:
            return

        try:
            task.status = 'processing'
            db.session.commit()

            # Run inference
            print(f"Starting inference for task {task_id} on file {file_path}")
            polygons = inference_service.predict(file_path)
            
            # Save results
            for poly_wkt in polygons:
                landslide = Landslide(
                    task_id=task.id,
                    geometry_wkt=poly_wkt,
                    area=0.0, # Calculate real area if pixel_size is known
                    confidence=0.9 # Placeholder, model outputs binary mask mostly
                )
                db.session.add(landslide)
            
            task.status = 'completed'
            task.completed_at = datetime.utcnow()
            db.session.commit()
            print(f"Task {task_id} completed. Found {len(polygons)} landslides.")

        except Exception as e:
            print(f"Task {task_id} failed: {e}")
            task.status = 'failed'
            db.session.commit()

@inference_bp.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        save_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
        file.save(save_path)
        
        # Create DB record
        image = SatelliteImage(
            filename=filename,
            file_path=save_path,
            satellite_source='Sentinel-2', # Default or from form data
            acquisition_date=datetime.utcnow() # Default
        )
        db.session.add(image)
        db.session.commit()
        
        return jsonify({
            'message': 'File uploaded successfully',
            'image_id': image.id,
            'filename': filename
        }), 201
    
    return jsonify({'error': 'File type not allowed'}), 400

@inference_bp.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    image_id = data.get('image_id')
    
    image = SatelliteImage.query.get(image_id)
    if not image:
        return jsonify({'error': 'Image not found'}), 404
    
    # Create Task
    task = DetectionTask(
        image_id=image.id,
        model_name='HSC_HENet',
        status='pending'
    )
    db.session.add(task)
    db.session.commit()
    
    # Start inference in background thread (Simple async for demo)
    # For production, use Celery
    thread = threading.Thread(target=run_inference_task, args=(task.id, image.file_path))
    thread.start()
    
    return jsonify({
        'message': 'Prediction task started',
        'task_id': task.id
    }), 202

@inference_bp.route('/tasks/<int:task_id>', methods=['GET'])
def get_task_status(task_id):
    task = DetectionTask.query.get(task_id)
    if not task:
        return jsonify({'error': 'Task not found'}), 404
    
    result = {
        'task_id': task.id,
        'status': task.status,
        'created_at': task.created_at,
        'completed_at': task.completed_at,
        'landslides_count': len(task.landslides) if task.status == 'completed' else 0
    }
    return jsonify(result)

@inference_bp.route('/results/<int:task_id>', methods=['GET'])
def get_results(task_id):
    task = DetectionTask.query.get(task_id)
    if not task or task.status != 'completed':
        return jsonify({'error': 'Task not completed or not found'}), 404
    
    features = []
    for landslide in task.landslides:
        features.append({
            'type': 'Feature',
            'geometry': {
                'wkt': landslide.geometry_wkt
                # Note: Frontend might need GeoJSON geometry object, 
                # but WKT is good for storage. Convert if needed.
            },
            'properties': {
                'id': landslide.id,
                'confidence': landslide.confidence,
                'area': landslide.area
            }
        })
    
    return jsonify({
        'type': 'FeatureCollection',
        'features': features
    })
