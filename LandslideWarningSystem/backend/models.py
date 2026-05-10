from datetime import datetime
from extensions import db

class User(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), unique=True, index=True)
    password_hash = db.Column(db.String(128))
    role = db.Column(db.String(20), default='inspector')  # 'admin' or 'inspector'

class Pipeline(db.Model):
    __tablename__ = 'pipelines'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100))
    geojson_path = db.Column(db.String(255))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class SatelliteImage(db.Model):
    __tablename__ = 'satellite_images'
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255))
    file_path = db.Column(db.String(255))
    satellite_source = db.Column(db.String(50)) # Sentinel-2, etc.
    acquisition_date = db.Column(db.DateTime)
    upload_time = db.Column(db.DateTime, default=datetime.utcnow)

class DetectionTask(db.Model):
    __tablename__ = 'detection_tasks'
    id = db.Column(db.Integer, primary_key=True)
    image_id = db.Column(db.Integer, db.ForeignKey('satellite_images.id'))
    model_name = db.Column(db.String(50), default='HSC_HENet')
    status = db.Column(db.String(20), default='pending') # pending, processing, completed, failed
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    completed_at = db.Column(db.DateTime)
    
    image = db.relationship('SatelliteImage', backref='tasks')

class Landslide(db.Model):
    __tablename__ = 'landslides'
    id = db.Column(db.Integer, primary_key=True)
    task_id = db.Column(db.Integer, db.ForeignKey('detection_tasks.id'))
    geometry_wkt = db.Column(db.Text) # Store Polygon WKT
    area = db.Column(db.Float)
    confidence = db.Column(db.Float)
    
    task = db.relationship('DetectionTask', backref='landslides')

class Alert(db.Model):
    __tablename__ = 'alerts'
    id = db.Column(db.Integer, primary_key=True)
    landslide_id = db.Column(db.Integer, db.ForeignKey('landslides.id'))
    pipeline_id = db.Column(db.Integer, db.ForeignKey('pipelines.id'), nullable=True)
    risk_level = db.Column(db.String(20)) # High, Medium, Low
    status = db.Column(db.String(20), default='unread') # unread, confirmed, resolved
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    landslide = db.relationship('Landslide', backref='alerts')
    pipeline = db.relationship('Pipeline', backref='alerts')
