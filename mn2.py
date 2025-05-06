import cv2
from flask import Flask, Response
import RPi.GPIO as GPIO
import time
import numpy as np
import sys # Import sys to get python version info if needed

from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort

def initialize_tracker():
    # print("Initializing DeepSORT tracker...") # Added debug print
    cfg = get_config()
    try:
        cfg.merge_from_file("deep_sort_pytorch/configs/deep_sort.yaml")
        # print("DeepSORT config loaded.") # Added debug print
    except Exception as e:
        print(f"Error loading DeepSORT config file: {e}")
        print("Please ensure 'deep_sort_pytorch/configs/deep_sort.yaml' exists and is accessible.")
        sys.exit(1) # Exit if config loading fails

    deepsort = DeepSort(
        cfg.DEEPSORT.REID_CKPT,
        max_dist=cfg.DEEPSORT.MAX_DIST,
        min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
        nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP,
        max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
        max_age=cfg.DEEPSORT.MAX_AGE,
        n_init=cfg.DEEPSORT.N_INIT,
        nn_budget=cfg.DEEPSORT.NN_BUDGET,
        use_cuda=False # Ensure this is False for Raspberry Pi
    )
    # print("DeepSORT tracker initialized successfully.") # Added debug print
    return deepsort

# GPIO setup
# print("Setting up GPIO...") # Added debug print
GPIO.setmode(GPIO.BCM)
LED_pin = 2 # Example pin, adjust if needed
GPIO.setup(LED_pin, GPIO.OUT)
GPIO.output(LED_pin, GPIO.LOW)
# print(f"GPIO pin {LED_pin} configured for LED output.") # Added debug print

app = Flask(__name__)

# Model and class file paths
MODEL_PB = "/home/zoqtmxhs/Desktop/capstone/frozen_inference_graph.pb"
MODEL_PBTXT = "/home/zoqtmxhs/Desktop/capstone/ssd_mobilenet_v2_coco_2018_03_29.pbtxt"
CLASSES_FILE = "/home/zoqtmxhs/Desktop/capstone/object_detection_classes_coco.txt"

# Load SSD-MobileNet model
try:
    # print(f"Attempting to load model from: {MODEL_PB}, {MODEL_PBTXT}") # Added debug print
    net = cv2.dnn.readNetFromTensorflow(MODEL_PB, MODEL_PBTXT)
    print("SSD-MobileNet model loaded successfully.")
except Exception as e:
    print(f"Error loading SSD-MobileNet model: {e}")
    print("Please check the paths to your .pb and .pbtxt files and file permissions.")
    sys.exit(1) # Exit if model loading fails

# Load class names
try:
    # print(f"Attempting to load classes from: {CLASSES_FILE}") # Added debug print
    with open(CLASSES_FILE, 'r') as f:
        CLASSES = [line.strip() for line in f.readlines()]
    print(f"{len(CLASSES)} class names loaded successfully.")
except Exception as e:
    print(f"Error loading class names file: {e}")
    print("Please check the path to your class names file and file permissions.")
    sys.exit(1) # Exit if class loading fails

# Define target classes and their IDs
TARGET_CLASSES = ['car', 'motorcycle', 'bicycle']
# Ensure class names exist before finding index
TARGET_CLASS_IDS = [CLASSES.index(cls) for cls in TARGET_CLASSES if cls in CLASSES]
if len(TARGET_CLASS_IDS) != len(TARGET_CLASSES):
    missing = [cls for cls in TARGET_CLASSES if cls not in CLASSES]
    print(f"Warning: Some target classes were not found in the loaded classes file: {missing}")

print(f"Target Class Names for LED control: {TARGET_CLASSES}")
print(f"Corresponding Class IDs: {TARGET_CLASS_IDS}")

CONFIDENCE_THRESHOLD = 0.3

# Initialize DeepSORT tracker
try:
    tracker = initialize_tracker()
    print("DeepSORT tracker initialized.")
except Exception as e:
    print(f"Failed to initialize DeepSORT tracker: {e}")
    sys.exit(1) # Exit if tracker initialization fails


# Camera setup
camera_path = "/dev/video4"
# print(f"Attempting to open camera device: {camera_path}") # Added debug print
cap = cv2.VideoCapture(camera_path, cv2.CAP_V4L2)

if cap.isOpened():
    print(f"Successfully opened camera device {camera_path}.")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    # print("Camera properties set.") # Added debug print
else:
    print(f"Cannot open the camera device {camera_path}. Trying default camera (index 0)...")
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    if cap.isOpened():
        print("Opened default camera (index 0) instead.")
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPEG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        # print("Camera properties set for default camera.") # Added debug print
    else:
        print("Cannot open default camera (index 0) either. Exiting.")
        sys.exit(1) # Exit if camera opening fails

def generate_frames():
    print("Starting frame generation...") # Added debug print
    frame_count = 0 # Added frame counter

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print(f"[{frame_count}] Failed to grab frame, retrying...") # Added frame_count
                time.sleep(0.1)
                continue

            frame_count += 1 # Increment frame counter
            # print(f"[{frame_count}] Frame captured. Shape: {frame.shape}") # Added debug print

            h, w, _ = frame.shape

            # Object Detection (SSD-MobileNet)
            blob = cv2.dnn.blobFromImage(frame, size=(300, 300), swapRB=True, crop=False)
            net.setInput(blob)
            detections = net.forward()
            # print(f"[{frame_count}] Detection forward pass completed. Shape: {detections.shape}") # Added debug print


            current_frame_detections = []
            current_frame_has_target = False # Flag for LED control

            detections = detections[0, 0] # Flatten the detections array

            # Process raw detections
            # print(f"[{frame_count}] Processing raw detections...") # Added debug print
            for detection in detections:
                confidence = detection[2]
                class_id = int(detection[1])

                if confidence > CONFIDENCE_THRESHOLD and class_id < len(CLASSES):
                     # Calculate pixel coordinates
                    x1 = int(detection[3] * w)
                    y1 = int(detection[4] * h)
                    x2 = int(detection[5] * w)
                    y2 = int(detection[6] * h)

                    # Validate detection bounding box
                    if x2 > x1 and y2 > y1:
                        current_frame_detections.append([x1, y1, x2, y2, confidence, class_id])
                        # print(f"[{frame_count}] Found detection: Class ID {class_id}, Confidence {confidence:.2f}, Bbox ({x1}, {y1}, {x2}, {y2})") # Added debug print

                        # Check if detected object is a target class for LED
                        if class_id in TARGET_CLASS_IDS:
                            current_frame_has_target = True

            # print(f"[{frame_count}] Processed {len(current_frame_detections)} valid detections.") # Added debug print

            # Update DeepSORT tracker
            # DeepSORT expects [x1, y1, x2, y2, confidence, class_id] or similar
            if current_frame_detections:
                detections_np = np.array(current_frame_detections)

                bbox_xyxy = detections_np[:, :4]
                confidences = detections_np[:, 4]
                classes = detections_np[:, 5] # Pass class IDs to the tracker

                # print(f"[{frame_count}] Updating tracker with {len(current_frame_detections)} detections.") # Added debug print
                tracked_objects = tracker.update(bbox_xyxy, confidences, classes, frame)
                # print(f"[{frame_count}] Tracker updated. Received {len(tracked_objects)} tracked objects.") # Added debug print

            else:
                 # Update tracker with empty detections if none found
                # print(f"[{frame_count}] No detections to update tracker with.") # Added debug print
                tracked_objects = tracker.update(np.empty((0, 4)), np.empty(0), np.empty(0), frame)
                # print(f"[{frame_count}] Tracker updated with no detections. Received {len(tracked_objects)} tracked objects.") # Added debug print


            # Control LED based on detections (not tracks)
            if current_frame_has_target:
                GPIO.output(LED_pin, GPIO.HIGH)
                # print(f"[{frame_count}] Target class detected. LED ON.") # Added debug print
            else:
                GPIO.output(LED_pin, GPIO.LOW)
                # print(f"[{frame_count}] No target class detected. LED OFF.") # Added debug print

            # Drawing tracked objects
            annotated_frame = frame.copy()

            # --- START: Enhanced Debugging for Drawing ---
            # 추적기에서 반환된 원본 tracked_objects 데이터를 출력합니다.
            print(f"[{frame_count}] Tracker Output (tracked_objects): {tracked_objects}")
            # --- END: Enhanced Debugging for Drawing ---


            for track_info in tracked_objects:
                 # DeepSORT 출력 형식은 일반적으로 [x1, y1, x2, y2, track_id, class_id] 입니다.
                 # track_info가 예상 형식을 가졌는지 확인합니다.
                 if isinstance(track_info, (list, np.ndarray)) and len(track_info) >= 6:
                    # 좌표, track_id, class_id를 추출합니다.
                    # 좌표를 그림 그리기에 필요한 정수로 변환합니다.
                    x1_float, y1_float, x2_float, y2_float = track_info[:4]
                    x1 = int(x1_float)
                    y1 = int(y1_float)
                    x2 = int(x2_float)
                    y2 = int(y2_float)
                    track_id = int(track_info[4])
                    # class_id는 추적 업데이트 시 검출기에서 받은 값을 사용합니다.
                    class_id_from_track = int(track_info[5])

                    # --- START: More Detailed Drawing Debugging ---
                    # 원본 부동 소수점 좌표와 정수 변환 후 좌표를 출력합니다.
                    print(f"[{frame_count}] Track ID {track_id}: Raw Coords = ({x1_float:.2f}, {y1_float:.2f}, {x2_float:.2f}, {y2_float:.2f})")
                    print(f"[{frame_count}] Track ID {track_id}: Int Coords = ({x1}, {y1}, {x2}, {y2})")
                    # 바운딩 박스가 유효한지 (x2 > x1 이고 y2 > y1 인지) 확인합니다.
                    is_valid_bbox = x2 > x1 and y2 > y1

                    if not is_valid_bbox:
                        # 바운딩 박스가 유효하지 않으면 그리기를 건너뛴다고 출력합니다.
                        print(f"[{frame_count}] Track ID {track_id}: Skipping drawing due to invalid bbox (x2 <= x1 or y2 <= y1)")
                        continue # 이 추적 대상은 그리기를 건너뜝니다.
                    else:
                         # 바운딩 박스가 유효하면 그리는 중이라고 출력합니다.
                         print(f"[{frame_count}] Track ID {track_id}: Bbox is valid. Drawing.")

                    # --- END: More Detailed Drawing Debugging ---

                    # 클래스 이름을 가져옵니다.
                    class_name = "Unknown" # 기본 클래스 이름
                    if 0 <= class_id_from_track < len(CLASSES):
                        class_name = CLASSES[class_id_from_track]
                    else:
                        # 클래스 ID가 범위를 벗어나는 경우를 처리합니다.
                        class_name = f"Class {class_id_from_track}"
                        print(f"[{frame_count}] Warning: Track ID {track_id} has unexpected class ID {class_id_from_track}.")


                    # 색상 정의 (녹색)
                    color = (0, 255, 0)

                    # 바운딩 박스 사각형을 그립니다.
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

                    # 텍스트 라벨을 준비합니다.
                    label = f"{class_name} ID: {track_id}"

                    # 텍스트 배경 사각형 크기를 가져옵니다.
                    (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    # 텍스트 위치를 바운딩 박스 바로 위로 설정합니다.
                    text_x = x1
                    text_y = y1 - baseline
                    # 텍스트가 프레임 위로 넘어가지 않도록 조정합니다.
                    if text_y < text_height:
                         text_y = y1 + text_height

                    # 텍스트 배경 사각형을 그립니다.
                    cv2.rectangle(annotated_frame, (text_x, text_y - text_height), (text_x + text_width, text_y), color, -1)
                    # 텍스트를 그립니다.
                    cv2.putText(annotated_frame, label, (text_x, text_y - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

                 # --- START: Debugging for unexpected track_info format ---
                 # 예상치 못한 track_info 형식이 들어온 경우 경고를 출력합니다.
                 else:
                     print(f"[{frame_count}] Warning: Unexpected track_info format: {track_info}")
                 # --- END: Debugging for unexpected track_info format ---


            # 프레임을 JPEG로 인코딩합니다.
            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            if not ret:
                print(f"[{frame_count}] Failed to encode frame.") # Added frame_count
                continue

            frame_bytes = buffer.tobytes()

            # Flask 스트림을 위해 프레임을 Yield 합니다.
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    except Exception as e:
        print(f"Error in generate_frames: {e}")
        # 에러 발생 시 에러 프레임을 Yield 하거나 스트림을 종료하는 로직을 추가할 수 있습니다.
        # yield (b'--frame\r\n' b'Content-Type: text/plain\r\n\r\nError processing frames: ' + str(e).encode() + b'\r\n')

    finally:
        print("generate_frames generator finished or an error occurred.")


@app.route('/video_feed')
def video_feed():
    # print("Video feed requested.") # Added debug print
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    # print("Index page requested.") # Added debug print
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
        # 필요에 따라 threaded=True로 설정하여 여러 클라이언트 처리를 개선할 수 있지만, 복잡성이 증가할 수 있습니다.
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=False)

    except KeyboardInterrupt:
        print("\nServer stopped by user (KeyboardInterrupt).")

    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        # traceback.print_exc() # 전체 트레이스백을 보려면 주석을 해제하세요.

    finally:
        print("Cleaning up resources...")
        # 카메라 객체가 존재하고 열려있는지 확인 후 해제합니다.
        if 'cap' in locals() and cap.isOpened():
             cap.release()
             print("Camera released.")
        elif 'cap' in locals():
             print("Camera object exists but was not successfully opened.")
        else:
             print("Camera object was not created.")

        # GPIO가 초기화되었는지 확인 후 정리합니다.
        if GPIO.getmode() is not None:
             GPIO.cleanup()
             print("GPIO cleaned up.")
        else:
             print("GPIO was not initialized.")

        print("Script finished.")
