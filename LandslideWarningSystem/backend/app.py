import os
from flask import Flask
from config import config
from extensions import db, cors

def create_app(config_name='default'):
    app = Flask(__name__)
    app.config.from_object(config[config_name])

    # Initialize extensions
    db.init_app(app)
    cors.init_app(app)

    # Register Blueprints
    from api.inference_routes import inference_bp
    app.register_blueprint(inference_bp, url_prefix='/api/analysis')

    # Ensure upload folder exists
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])

    @app.route('/')
    def index():
        return "Landslide Warning System API is Running!"

    return app

if __name__ == '__main__':
    app = create_app(os.getenv('FLASK_CONFIG') or 'default')
    with app.app_context():
        db.create_all() # Create tables for development
    app.run(host='0.0.0.0', port=5000)
