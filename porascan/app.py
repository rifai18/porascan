from flask import Flask, render_template, request
import os
import uuid

from utils.yolo import detect_image
from utils.gemini import generate_explanation
from utils.db import db, cursor

from flask import redirect

from flask import jsonify
import base64
import cv2
import numpy as np

from utils.yolo import detect_realtime

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/history')
def history():

    cursor.execute("""
        SELECT * FROM detections
        ORDER BY id DESC
    """)

    data = cursor.fetchall()

    return render_template(
        'history.html',
        data=data
    )

@app.route('/delete-history/<int:id>')
def delete_history(id):

    cursor.execute("""
        SELECT result_image
        FROM detections
        WHERE id = %s
    """, (id,))

    data = cursor.fetchone()

    if data:

        image_path = os.path.join(
            "static/results",
            data[0]
        )

        if os.path.exists(image_path):
            os.remove(image_path)

    cursor.execute("""
        DELETE FROM detections
        WHERE id = %s
    """, (id,))

    db.commit()

    return redirect('/history')

@app.route('/live')
def live():
    return render_template('live.html')

@app.route('/detect', methods=['POST'])
def detect():

    file = request.files['image']

    filename = f"{uuid.uuid4()}.jpg"

    filepath = os.path.join(
        UPLOAD_FOLDER,
        filename
    )

    file.save(filepath)

    detection = detect_image(filepath)

    all_diseases = ", ".join([
        f"{item['disease']} ({item['confidence']}%)"
        for item in detection['detections']
    ])

    explanation = generate_explanation(all_diseases)

    sql = """
    INSERT INTO detections
    (disease_name, confidence, original_image, result_image, ai_explanation)
    VALUES (%s, %s, %s, %s, %s)
    """

    values = (
        all_diseases,
        0,
        filename,
        detection['result_image'],
        explanation
    )

    cursor.execute(sql, values)

    db.commit()

    return render_template(
        'result.html',
        detections=detection['detections'],
        result_image=detection['result_image'],
        explanation=explanation
    )

@app.route('/live-detect', methods=['POST'])
def live_detect():

    file = request.files['image']

    file_bytes = np.frombuffer(
        file.read(),
        np.uint8
    )

    frame = cv2.imdecode(
        file_bytes,
        cv2.IMREAD_COLOR
    )

    detections = detect_realtime(frame)

    return jsonify({
        "detections": detections
    })

if __name__ == '__main__':
    app.run(debug=True)