# app/ndvi.py
import cv2
import numpy as np

def align_images(base_img, img_to_align, max_features=400, good_match_percent=0.15):
    """
    Align img_to_align to base_img using ORB feature matching and homography.
    If matching fails or too few matches, will fall back to simple center-crop/resize.
    Returns aligned image (same size as base_img).
    """
    try:
        if base_img is None or img_to_align is None:
            raise ValueError("One or both images are None")
        im1_gray = cv2.cvtColor(base_img, cv2.COLOR_BGR2GRAY)
        im2_gray = cv2.cvtColor(img_to_align, cv2.COLOR_BGR2GRAY)

        orb = cv2.ORB_create(max_features)
        kp1, des1 = orb.detectAndCompute(im1_gray, None)
        kp2, des2 = orb.detectAndCompute(im2_gray, None)

        if des1 is None or des2 is None or len(kp1) < 8 or len(kp2) < 8:
            # Not enough features to match
            raise RuntimeError("Insufficient keypoints/descriptors for ORB alignment")

        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = matcher.match(des1, des2)
        if len(matches) < 8:
            raise RuntimeError("Too few matches")

        matches = sorted(matches, key=lambda x: x.distance)
        numGood = max(4, int(len(matches) * good_match_percent))
        matches = matches[:numGood]

        pts1 = np.zeros((len(matches), 2), dtype=np.float32)
        pts2 = np.zeros((len(matches), 2), dtype=np.float32)

        for i, m in enumerate(matches):
            pts1[i, :] = kp1[m.queryIdx].pt
            pts2[i, :] = kp2[m.trainIdx].pt

        h, mask = cv2.findHomography(pts2, pts1, cv2.RANSAC)
        if h is None:
            raise RuntimeError("Homography computation failed")

        height, width = base_img.shape[:2]
        aligned = cv2.warpPerspective(img_to_align, h, (width, height))
        return aligned
    except Exception:
        # fallback: resize img_to_align to base_img size (simple)
        try:
            h, w = base_img.shape[:2]
            resized = cv2.resize(img_to_align, (w, h))
            return resized
        except Exception:
            # as last resort, return base_img itself
            return base_img.copy()

def compute_ndvi(nir_bgr, rgb_bgr, clip_percent=1):
    """
    Compute NDVI given NIR image (BGR where RED channel contains NIR) and RGB image.
    Returns: (ndvi_float_map, ndvi_color_vis_uint8)
    ndvi_float_map: values in [-1,1]
    ndvi_color_vis_uint8: color mapped uint8 image for visualization
    """
    if nir_bgr is None or rgb_bgr is None:
        raise ValueError("Input images cannot be None")
    # ensure same size
    if nir_bgr.shape != rgb_bgr.shape:
        rgb_bgr = cv2.resize(rgb_bgr, (nir_bgr.shape[1], nir_bgr.shape[0]))
    # extract channels (cv2 uses BGR)
    nir = nir_bgr[:, :, 2].astype(np.float32)
    red = rgb_bgr[:, :, 2].astype(np.float32)
    denom = (nir + red)
    denom[denom == 0] = 1e-6
    ndvi = (nir - red) / denom
    # clip extremes for visualization
    low, high = np.percentile(ndvi.flatten(), [clip_percent, 100 - clip_percent])
    ndvi_clipped = np.clip(ndvi, low, high)
    # normalize to 0-255 for visualization
    ndvi_norm = ((ndvi_clipped - ndvi_clipped.min()) / (ndvi_clipped.max() - ndvi_clipped.min() + 1e-9) * 255.0).astype(np.uint8)
    ndvi_color = cv2.applyColorMap(ndvi_norm, cv2.COLORMAP_JET)
    return ndvi, ndvi_color
