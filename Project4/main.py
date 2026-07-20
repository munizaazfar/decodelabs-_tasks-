import cv2
import numpy as np
import os

# PASCAL VOC Dataset ke 21 classes
CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
    "sofa", "train", "tvmonitor"
]

# Random colors for bounding boxes
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

def run_object_detection(
    image_path: str = "test_image.jpg",
    prototxt_path: str = "MobileNetSSD_deploy.prototxt",
    model_path: str = "MobileNetSSD_deploy.caffemodel",
    confidence_gate: float = 0.80
):
    if not os.path.exists(image_path):
        print(f"Error: '{image_path}' nahi mili! Pehly ek image 'Project4' folder me rakhein aur naam 'test_image.jpg' rakhein.")
        return

    # 1. Ingest Image Matrix
    image = cv2.imread(image_path)
    (h, w) = image.shape[:2]

    # 2. Load Neural Network
    net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

    # 3. Pre-Processing: 4D Blob Construction (300x300, scale factor 0.007843, mean 127.5)
    blob = cv2.dnn.blobFromImage(
        cv2.resize(image, (300, 300)),
        scalefactor=0.007843,
        size=(300, 300),
        mean=127.5
    )

    # 4. Forward Pass
    net.setInput(blob)
    detections = net.forward()

    valid_detections = 0
    
    # 5. Apply 80% Confidence Gate
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence >= confidence_gate:
            valid_detections += 1
            idx = int(detections[0, 0, i, 1])
            
            # Map normalized coordinates back to actual pixel dimensions
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            label_text = f"{CLASSES[idx]}: {confidence * 100:.2f}%"
            color = COLORS[idx]

            # Bounding Box and Label Drawing
            cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)
            y_pos = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(
                image, label_text, (startX, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
            )
            print(f"✓ Detected: {CLASSES[idx]} | Confidence: {confidence*100:.2f}%")

    # 6. Output Save
    output_path = "output_detection_result.png"
    cv2.imwrite(output_path, image)
    
    print("\n--- OBJECT DETECTION COMPLETE ---")
    print(f"Total Validated Objects (>= 80%): {valid_detections}")
    print(f"Result Image Saved As: {output_path}")

if __name__ == "__main__":
    run_object_detection(confidence_gate=0.50)