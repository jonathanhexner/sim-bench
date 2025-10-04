"""
Earth Mover's Distance (EMD) method for HSV histograms.
Uses Strategy pattern for distance computation.
"""

import cv2
import numpy as np
from typing import List
from sim_bench.methods.base import BaseMethod


class EMDMethod(BaseMethod):
    """HSV histograms with configurable distance strategy."""
    
    def _preprocess(self, img, resize=(256,256), center_crop=(224,224)):
        """Preprocess image with resize and center crop."""
        if resize:
            img = cv2.resize(img, resize, interpolation=cv2.INTER_AREA)
        if center_crop:
            h, w = img.shape[:2]
            ch, cw = center_crop
            y0 = max(0, (h - ch) // 2)
            x0 = max(0, (w - cw) // 2)
            img = img[y0:y0+ch, x0:x0+cw]
        return img
    
    def extract_features(self, image_paths: List[str]) -> np.ndarray:
        """Extract HSV histograms from images."""
        bins = tuple(self.config['features']['bins'])
        preproc = self.config['features'].get('preproc', {})
        
        print(f"Extracting HSV histograms ({bins} bins)...")
        
        H = []
        for f in image_paths:
            img = cv2.imread(f)
            if img is None:
                raise FileNotFoundError(f)
            if preproc:
                img = self._preprocess(
                    img,
                    resize=tuple(preproc.get('resize')) if preproc.get('resize') else None,
                    center_crop=tuple(preproc.get('center_crop')) if preproc.get('center_crop') else None
                )
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0,1,2], None, bins, [0,180,0,256,0,256])
            hist = cv2.normalize(hist, hist).flatten()
            H.append(hist.astype('float32'))
        return np.vstack(H)
