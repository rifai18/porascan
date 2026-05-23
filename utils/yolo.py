from importlib.resources import files

from ultralytics import YOLO
import os
import uuid
import shutil
import cv2
import numpy as np

model = YOLO("model/best.pt")

def detect_realtime(frame):

    results = model.predict(
        source=frame,
        imgsz=256,
        conf=0.4,
        save=False,
        verbose=False,
        device='cpu'
    )

    r = results[0]

    detections = []

    if r.boxes is not None:

        boxes = r.boxes.xyxy.cpu().numpy().tolist()

        cls_values = r.boxes.cls

        conf_values = r.boxes.conf

        for cls, conf, box in zip(
            cls_values,
            conf_values,
            boxes
        ):

            detections.append({
                "label": r.names[int(cls)],
                "confidence": float(conf),
                "box": box
            })

    return detections

def detect_image(image_path):

    results = model.predict(
        source=image_path,
        save=False,
        verbose=False
    )

    r = results[0]

    detections = []

    if r.boxes is not None:

        for i in range(len(r.boxes.cls)):

            cls_id = int(r.boxes.cls[i])

            conf = float(r.boxes.conf[i])

            disease = r.names[cls_id]

            detections.append({
                "disease": disease,
                "confidence": round(conf * 100, 2)
            })

    plotted = r.plot()

    os.makedirs("static/results", exist_ok=True)

    final_name = f"{uuid.uuid4()}.jpg"

    final_path = os.path.join(
        "static/results",
        final_name
    )

    cv2.imwrite(final_path, plotted)

    return {
        "detections": detections,
        "result_image": final_name
    }