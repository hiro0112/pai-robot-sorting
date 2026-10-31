"""Synthetic camera samples and supervised classification for a sorting robot.

All imagery and labels are original; no externally supplied training data.
"""
from pathlib import Path
import cv2
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

CLASSES = ('MEDICINE', 'WATER', 'FOOD')
BASE_RGB = {
    'MEDICINE': (238, 91, 83),    # red rectangle
    'WATER': (85, 172, 237),      # blue circle
    'FOOD': (96, 208, 133),       # green triangle
}
BACKGROUND_BGR = np.array([40, 31, 22], dtype=np.uint8)


def sample_image(label, rng, size=96):
    """Render a simulated inspection camera image (BGR), with nuisance variation."""
    rgb = np.array(BASE_RGB[label], dtype=np.float32)
    rgb = np.clip(rgb + rng.normal(0, 25, 3), 48, 255).astype(np.uint8)
    bgr = tuple(int(t) for t in rgb[::-1])
    img = np.full((size, size, 3), BACKGROUND_BGR, dtype=np.uint8)
    center = (int(size/2 + rng.integers(-5, 6)), int(size/2 + rng.integers(-5, 6)))
    radius = int(rng.integers(23, 31))
    if label == 'MEDICINE':
        # Rotation creates meaningful visual variation while retaining four corners.
        rect = (center, (radius * 1.75, radius * 1.15), float(rng.uniform(-27, 27)))
        pts = np.int32(cv2.boxPoints(rect))
        cv2.fillConvexPoly(img, pts, bgr)
    elif label == 'WATER':
        cv2.circle(img, center, radius, bgr, -1, lineType=cv2.LINE_AA)
    elif label == 'FOOD':
        a = float(rng.uniform(-0.27, 0.27))
        points = []
        for k in range(3):
            t = a + 2*np.pi*k/3 - np.pi/2
            points.append([center[0] + radius*np.cos(t), center[1] + radius*np.sin(t)])
        cv2.fillConvexPoly(img, np.int32(points), bgr)
    noise = rng.normal(0, 5, img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32)+noise,0,255).astype(np.uint8)


def extract_features(img):
    """Feature extraction from RGB image, not privileged access to the true label."""
    arr = img.astype(np.float32)
    delta = np.linalg.norm(arr - BACKGROUND_BGR.astype(np.float32), axis=2)
    mask = (delta > 65).astype(np.uint8)*255
    contours,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError('No package detected in camera frame')
    c = max(contours, key=cv2.contourArea)
    fg = np.zeros(mask.shape, dtype=np.uint8)
    cv2.drawContours(fg, [c], -1, 255, -1)
    pixels = img[fg > 0].astype(np.float32)
    mean_rgb = pixels.mean(axis=0)[::-1]/255.0
    area = cv2.contourArea(c)
    perimeter = cv2.arcLength(c, True)
    circularity = 4*np.pi*area/max(perimeter*perimeter, 1e-10)
    vertex_count = len(cv2.approxPolyDP(c, 0.035*perimeter, True))
    return np.array([*mean_rgb, area/(img.shape[0]*img.shape[1]),
                     circularity, min(vertex_count,20)/20.0], dtype=np.float64)


def train_model(path, seed=2026, samples_per_class=220):
    rng = np.random.default_rng(seed)
    X,y = [],[]
    for i,label in enumerate(CLASSES):
        for _ in range(samples_per_class):
            X.append(extract_features(sample_image(label, rng)))
            y.append(i)
    X=np.stack(X)
    y=np.array(y)
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,
        stratify=y,random_state=seed)
    model=make_pipeline(StandardScaler(),MLPClassifier(hidden_layer_sizes=(28,14),
        max_iter=800,random_state=seed, early_stopping=False, n_iter_no_change=35))
    model.fit(X_train,y_train)
    pred=model.predict(X_test)
    stats={
        'seed':seed, 'training_samples':int(len(X_train)),
        'test_samples':int(len(X_test)),
        'test_accuracy':float(accuracy_score(y_test,pred)),
        'class_names':list(CLASSES),
        'confusion_matrix':confusion_matrix(y_test,pred).tolist(),
        'classification_report':classification_report(y_test,pred,target_names=CLASSES,output_dict=True),
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model,path)
    return model,stats


def infer(model,img):
    feature=extract_features(img).reshape(1,-1)
    probs=model.predict_proba(feature)[0]
    idx=int(np.argmax(probs))
    return CLASSES[idx], float(probs[idx]), probs.tolist()
