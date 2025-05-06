import cv2
from flask import Flask, Response
import RPi.GPIO as GPIO
import time
import numpy as np
import sys

from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort

def initialize_tracker():
    cfg = get_config()
    try:
        cfg.merge_from_file("deep_sort_pytorch/configs/deep_sort.yaml")
    except Exception as e:
        print(f"Error loading DeepSORT config file: {e}")
        print("Please ensure 'deep_sort_pytorch/configs/deep_sort.yaml' exists and is accessible.")
        sys.exit(1)

    deepsort = DeepSort(
        cfg.DEEPSORT.REID_CKPT,
        max_dist=cfg.DEEPSORT.MAX_DIST,
        min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
        nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP,
        max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
        max_age=cfg.DEEPSORT.MAX_AGE,
        n_init=cfg.DEEPSORT.N_INIT,
        nn_budget=cfg.DEEPSORT.NN_BUDGET,
        use_cuda=False
    )
    return deepsort

# GPIO setup
GPIO.setmode(GPIO.BCM)
LED_pin = 2
GPIO.setup(LED_pin, GPIO.OUT)
GPIO.output(LED_pin, GPIO.LOW)

app = Flask(__name__)

# Model and class file paths
MODEL_PB = "/home/zoqtmxhs/Desktop/capstone/frozen_inference_graph.pb"
MODEL_PBTXT = "/home/zoqtmxhs/Desktop/capstone/ssd_mobilenet_v2_coco_2018_03_29.pbtxt"
CLASSES_FILE = "/home/zoqtmxhs/Desktop/capstone/object_detection_classes_coco.txt"

# Load SSD-MobileNet model
try:
    net = cv2.dnn.readNetFromTensorflow(MODEL_PB, MODEL_PBTXT)
    print("SSD-MobileNet model loaded successfully.")
except Exception as e:
    print(f"Error loading SSD-MobileNet model: {e}")
    print("Please check the paths to your .pb and .pbtxt files and file permissions.")
    sys.exit(1)

# Load class names
try:
    with open(CLASSES_FILE, 'r') as f:
        CLASSES = [line.strip() for line in f.readlines()]
    print(f"{len(CLASSES)} class names loaded successfully.")
    # --- DEBUG PRINT: Loaded Classes List ---
    print("Loaded CLASSES:", CLASSES)
    # --- END DEBUG PRINT ---
except Exception as e:
    print(f"Error loading class names file: {e}")
    print("Please check the path to your class names file and file permissions.")
    sys.exit(1)

# Define target classes and their IDs
TARGET_CLASSES = ['car', 'motorcycle', 'bicycle', 'person']

# Ensure class names exist before finding index
TARGET_CLASS_IDS = []
for cls in TARGET_CLASSES:
    try:
        TARGET_CLASS_IDS.append(CLASSES.index(cls))
    except ValueError:
        print(f"Warning: Target class '{cls}' not found in {CLASSES_FILE}. It will be ignored.")

if len(TARGET_CLASS_IDS) != len(TARGET_CLASSES):
    found_target_classes = [CLASSES[i] for i in TARGET_CLASS_IDS]
    missing = [cls for cls in TARGET_CLASSES if cls not in found_target_classes]
    if missing:
         print(f"Warning: Some target classes were not found in the loaded classes file: {missing}")


print(f"Target Class Names for LED control: {TARGET_CLASSES}")
# --- DEBUG PRINT: Calculated Target Class IDs ---
print("Calculated TARGET_CLASS_IDS:", TARGET_CLASS_IDS)
# --- END DEBUG PRINT ---


CONFIDENCE_THRESHOLD = 0.3

# --- NEW SETTING ---
# Set to True to only feed detections of TARGET_CLASSES to the tracker
# Set to False to feed all detections above CONFIDENCE_THRESHOLD to the tracker
FILTER_DETECTIONS_BY_TARGET_CLASSES = True # <--- 이 값을 True로 유지하세요. 관심 객체만 추적/표시합니다.
# --- END NEW SETTING ---

# Initialize DeepSORT tracker
try:
    tracker = initialize_tracker()
    print("DeepSORT tracker initialized.")
except Exception as e:
    print(f"Failed to initialize DeepSORT tracker: {e}")
    sys.exit(1)


# Camera setup
camera_path = "/dev/video4"
cap = cv2.VideoCapture(camera_path, cv2.CAP_V4L2)

if cap.isOpened():
    print(f"Successfully opened camera device {camera_path}.")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
else:
    print(f"Cannot open the camera device {camera_path}. Trying default camera (index 0)...")
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    if cap.isOpened():
        print("Opened default camera (index 0) instead.")
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    else:
        print("Cannot open default camera (index 0) either. Exiting.")
        sys.exit(1)

def generate_frames():
    print("Starting frame generation...")
    frame_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print(f"[{frame_count}] Failed to grab frame, retrying...")
                time.sleep(0.1)
                continue

            frame_count += 1
            h, w, _ = frame.shape

            blob = cv2.dnn.blobFromImage(frame, size=(300, 300), swapRB=True, crop=False)
            net.setInput(blob)
            detections = net.forward()

            # List to hold detections for THIS frame that meet confidence threshold
            # This list contains ALL valid detections before class filtering for tracking
            all_valid_detections_for_frame = []

            # Flag for LED control (based on ANY target class detection from ALL valid detections)
            current_frame_has_target = False

            detections = detections[0, 0]

            # Process raw detections from the model
            for detection in detections:
                confidence = detection[2]
                class_id = int(detection[1]) # Model's predicted class ID

                # Check if confidence is high enough and class_id is valid based on our CLASSES list size
                if confidence > CONFIDENCE_THRESHOLD and class_id < len(CLASSES):
                    x1 = int(detection[3] * w)
                    y1 = int(detection[4] * h)
                    x2 = int(detection[5] * w)
                    y2 = int(detection[6] * h)

                    # Ensure the bounding box is valid before adding
                    if x2 > x1 and y2 > y1:
                        # Add detection to the list of all valid detections for this frame
                        all_valid_detections_for_frame.append([x1, y1, x2, y2, confidence, class_id])

                        # Check if this detection is a target class for LED control
                        # (This check uses the raw class_id from the model)
                        if class_id in TARGET_CLASS_IDS:
                            current_frame_has_target = True

            # --- Apply detection filtering before passing to tracker ---
            # FILTER_DETECTIONS_BY_TARGET_CLASSES=True 이므로 TARGET_CLASSES에 있는 객체만 추적기에 전달됩니다.
            detections_to_track = []
            if FILTER_DETECTIONS_BY_TARGET_CLASSES:
                # Only pass detections whose class ID is in TARGET_CLASS_IDS to the tracker
                detections_to_track = [
                    det for det in all_valid_detections_for_frame
                    if det[5] in TARGET_CLASS_IDS # Filter based on class ID
                ]
                # print(f"[{frame_count}] Passing {len(detections_to_track)} TARGET detections to tracker.")
            else:
                # Pass ALL valid detections above confidence threshold to the tracker
                detections_to_track = all_valid_detections_for_frame
                # print(f"[{frame_count}] Passing {len(detections_to_track)} ALL valid detections to tracker.")
            # --- End of filtering ---

            # Update DeepSORT tracker with the filtered detections
            if detections_to_track:
                detections_np = np.array(detections_to_track)
                bbox_xyxy = detections_np[:, :4]
                confidences = detections_np[:, 4]
                classes = detections_np[:, 5] # Pass the *filtered* class IDs to the tracker

                tracked_objects = tracker.update(bbox_xyxy, confidences, classes, frame)
            else:
                # Update tracker with empty detections
                tracked_objects = tracker.update(np.empty((0, 4)), np.empty(0), np.empty(0), frame)


            # Control LED based on detections (still based on ANY target class detection found in ALL valid detections)
            if current_frame_has_target:
                GPIO.output(LED_pin, GPIO.HIGH)
            else:
                GPIO.output(LED_pin, GPIO.LOW)

            # Drawing tracked objects
            annotated_frame = frame.copy()

            # --- Debugging: Print tracker output format ---
            # print(f"[{frame_count}] Tracker Output (tracked_objects): {tracked_objects}")
            # --- End Debugging ---

            # --- Drawing loop FIX for ([], []) output ---
            tracked_items_to_draw = []
            if isinstance(tracked_objects, tuple) and len(tracked_objects) > 0:
                 potential_tracked_items = tracked_objects[0]
                 if isinstance(potential_tracked_items, (list, np.ndarray)):
                     tracked_items_to_draw = potential_tracked_items

            # Check if we got a valid list/array of tracked items AND if it's not empty
            if isinstance(tracked_items_to_draw, (list, np.ndarray)) and len(tracked_items_to_draw) > 0:
                for track_info in tracked_items_to_draw:
                    # track_info format from tracker update: [x1, y1, x2, y2, track_id, class_id]
                    # The class_id here is the one that was passed TO the tracker
                    if isinstance(track_info, (list, np.ndarray)) and len(track_info) >= 6:
                         x1_float, y1_float, x2_float, y2_float = track_info[:4]
                         x1 = int(x1_float)
                         y1 = int(y1_float)
                         x2 = int(x2_float)
                         y2 = int(y2_float)
                         track_id = int(track_info[4])
                         # Class ID associated with the track (this came from the filtered detection)
                         class_id_from_track = int(track_info[5])


                         is_valid_bbox = x2 > x1 and y2 > y1

                         if not is_valid_bbox:
                             # print(f"[{frame_count}] Track ID {track_id}: Skipping drawing due to invalid bbox. Coords: ({x1}, {y1}, {x2}, {y2})")
                             continue

                         # --- Debugging: Print info for tracks being drawn ---
                         # print(f"[{frame_count}] Drawing Track ID {track_id}, Class ID {class_id_from_track}, Coords ({x1}, {y1}, {x2}, {y2})")
                         # --- End Debugging ---

                         class_name = "Unknown"
                         # Get class name using the ID from the track_info (which came from filtered detections)
                         if 0 <= class_id_from_track < len(CLASSES):
                             class_name = CLASSES[class_id_from_track]
                         else:
                             class_name = f"Class {class_id_from_track}"
                             print(f"[{frame_count}] Warning: Track ID {track_id} has unexpected class ID {class_id_from_track} for drawing.")

                         color = (0, 255, 0) # Green

                         cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

                         label = f"{class_name} ID: {track_id}"

                         (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                         text_x = x1
                         text_y = y1 - baseline
                         if text_y < text_height:
                              text_y = y1 + text_height

                         cv2.rectangle(annotated_frame, (text_x, text_y - text_height), (text_x + text_width, text_y), color, -1)
                         cv2.putText(annotated_frame, label, (text_x, text_y - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

            # --- Warning for unexpected TOP-LEVEL formats ---
            elif not (isinstance(tracked_objects, tuple) and len(tracked_objects) > 0 and isinstance(tracked_objects[0], (list, np.ndarray))):
                 # Only print if tracked_objects is not None and not the expected ([], []) or (np.empty, []) format
                 if tracked_objects is not None and not (isinstance(tracked_objects, tuple) and len(tracked_objects) > 0 and (isinstance(tracked_objects[0], list) and len(tracked_objects[0]) == 0) or (isinstance(tracked_objects[0], np.ndarray) and tracked_objects[0].size == 0)):
                      print(f"[{frame_count}] Warning: tracked_objects has unexpected format: {tracked_objects}")
            # --- END Warning ---
            # --- END Drawing loop FIX ---


            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            if not ret:
                print(f"[{frame_count}] Failed to encode frame.")
                continue

            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    except Exception as e:
        print(f"Error in generate_frames: {e}")
    finally:
        print("generate_frames generator finished or an error occurred.")


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return """
    <html>
    <head>
    <title>Real-time SSD-MobileNet & DeepSORT Tracking</title>
    </head>
    <body>
    <h1>Live Camera Feed with SSD-MobileNet & DeepSORT Tracking</h1>
    <img src="/video_feed" width="640" height="480">
    </body>
    </html>
    """

if __name__ == "__main__":
    try:
        print("Starting Flask app...")
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=False)

    except KeyboardInterrupt:
        print("\nServer stopped by user (KeyboardInterrupt).")

    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")

    finally:
        print("Cleaning up resources...")
        if 'cap' in locals() and cap.isOpened():
             cap.release()
             print("Camera released.")
        elif 'cap' in locals():
             print("Camera object exists but was not successfully opened.")
        else:
             print("Camera object was not created.")

        if GPIO.getmode() is not None:
             GPIO.cleanup()
             print("GPIO cleaned up.")
        else:
             print("GPIO was not initialized.")

        print("Script finished.")
