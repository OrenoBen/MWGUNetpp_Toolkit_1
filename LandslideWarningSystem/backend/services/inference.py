import sys
import os
import torch
import numpy as np
import h5py
from shapely.geometry import Polygon, MultiPolygon
from shapely.wkt import dumps
from torchvision import transforms

# Ensure project root is in path to import src modules
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.models.HSC_HENet import HSC_HENet
from src.data.landslide4sense import LandslideDataset

class InferenceService:
    def __init__(self, model_path=None, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model(model_path)
        
        # Preprocessing transforms (same as training)
        self.transform = transforms.Compose([
            transforms.Resize((256, 256))
        ])

    def _load_model(self, model_path):
        # Initialize model architecture
        model = HSC_HENet(
            n_channels=14,
            n_classes=1,
            base_channels=64,
            deep_supervision=False 
        )
        
        if model_path and os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            # Handle potential key mismatch if saved with DataParallel or different structure
            state_dict = checkpoint.get("model", checkpoint)
            model.load_state_dict(state_dict, strict=False)
            print(f"Model loaded from {model_path}")
        else:
            print("Warning: No model path provided or file not found. Using initialized weights.")
        
        model.to(self.device)
        model.eval()
        return model

    def preprocess_image(self, file_path):
        """
        Load and preprocess the image from H5 file.
        Returns tensor ready for inference.
        """
        try:
            with h5py.File(file_path, "r") as f:
                # Assuming similar structure to LandslideDataset._load_h5
                # Try finding image data
                possible_keys = ["img", "image", "data", "array"]
                data = None
                for key in possible_keys:
                    if key in f:
                        data = f[key][:]
                        break
                
                if data is None:
                    # Fallback: take first key
                    keys = list(f.keys())
                    if keys:
                        data = f[keys[0]][:]
                    else:
                        raise ValueError("No data found in H5 file")

            # Standardization (Mean/Std from LandslideDataset)
            # Note: This should ideally match the training normalization exactly
            mean = np.array([423.1, 541.5, 629.8, 673.9, 832.4, 1648.2, 1874.2, 
                             1997.9, 2076.4, 2110.2, 2203.5, 2326.1, 1758.3, 956.2])
            std = np.array([85.2, 98.7, 112.3, 135.6, 156.8, 234.5, 289.1, 
                            312.4, 328.7, 335.1, 356.2, 389.7, 301.2, 218.5])
            
            # (H, W, C) -> (C, H, W)
            if data.shape[-1] == 14:
                data = np.transpose(data, (2, 0, 1))
            
            # Normalize
            for i in range(14):
                data[i] = (data[i] - mean[i]) / (std[i] + 1e-8)
            
            # Convert to tensor
            tensor = torch.from_numpy(data.copy()).float()
            
            # Resize
            tensor = self.transform(tensor)
            
            # Add batch dimension: (1, C, H, W)
            return tensor.unsqueeze(0)

        except Exception as e:
            print(f"Error preprocessing image: {e}")
            raise e

    def predict(self, file_path):
        """
        Run inference on a single file.
        Returns: list of polygon WKT strings representing detected landslides.
        """
        input_tensor = self.preprocess_image(file_path).to(self.device)
        
        with torch.no_grad():
            # Forward pass
            output = self.model(input_tensor)
            
            # Sigmoid activation -> Probability map
            prob_map = torch.sigmoid(output).squeeze().cpu().numpy() # (H, W)
            
            # Thresholding
            mask = (prob_map > 0.5).astype(np.uint8)
            
            # Post-processing: Extract polygons from mask
            polygons = self._mask_to_polygons(mask)
            
            return polygons

    def _mask_to_polygons(self, mask):
        """
        Convert binary mask to WKT polygons using Rasterio/Shapely logic (simplified here with opencv or manual contour finding if needed, 
        but referencing rasterio features.shapes is standard for geo-masks).
        
        Since we don't have geo-transform info in raw H5 usually, we return pixel coordinates.
        Real system would map these to Lat/Lon using metadata.
        """
        import cv2
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        wkt_list = []
        for contour in contours:
            if len(contour) < 3:
                continue
            
            # contour shape: (N, 1, 2) -> (N, 2)
            points = contour.reshape(-1, 2)
            
            # Create polygon
            poly = Polygon(points)
            
            # Simplify slightly to reduce complexity
            poly = poly.simplify(1.0, preserve_topology=True)
            
            if not poly.is_empty and poly.area > 10: # Filter tiny noise
                wkt_list.append(dumps(poly))
                
        return wkt_list

# Singleton instance (loaded on app start)
# In production, model path should be in config
DEFAULT_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "best_landslide_classifier.pth") 
inference_service = InferenceService(model_path=DEFAULT_MODEL_PATH)
