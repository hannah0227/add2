import cv2

from flask import Flask, Response

import RPi.GPIO as GPIO

import time

import numpy as np



from deep_sort_pytorch.utils.parser import get_config

from deep_sort_pytorch.deep_sort import DeepSort



def initialize_tracker():

    cfg = get_config()

    cfg.merge_from_file("deep_sort_pytorch/configs/deep_sort.yaml")



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



GPIO.setmode(GPIO.BCM)

LED_pin = 2

GPIO.setup(LED_pin, GPIO.OUT)

GPIO.output(LED_pin, GPIO.LOW)



app = Flask(__name__)



MODEL_PB = "/home/zoqtmxhs/Desktop/capstone/frozen_inference_graph.pb"

MODEL_PBTXT = "/home/zoqtmxhs/Desktop/capstone/ssd_mobilenet_v2_coco_2018_03_29.pbtxt"

CLASSES_FILE = "/home/zoqtmxhs/Desktop/capstone/object_detection_classes_coco.txt"



try:

    net = cv2.dnn.readNetFromTensorflow(MODEL_PB, MODEL_PBTXT)

    print("SSD-MobileNet model loaded successfully.")

except Exception as e:

    print(f"Error loading SSD-MobileNet model: {e}")

    print("Please check the paths to your .pb and .pbtxt files.")

    exit()



try:

    with open(CLASSES_FILE, 'r') as f:

        CLASSES = [line.strip() for line in f.readlines()]

    print(f"{len(CLASSES)} class names loaded successfully.")

except Exception as e:

    print(f"Error loading class names file: {e}")

    print("Please check the path to your class names file.")

    exit()



TARGET_CLASSES = ['car', 'motorcycle', 'bicycle']

TARGET_CLASS_IDS = [CLASSES.index(cls) for cls in TARGET_CLASSES if cls in CLASSES]

print(f"Target Class Names for LED control: {TARGET_CLASSES}")

print(f"Corresponding Class IDs: {TARGET_CLASS_IDS}")



CONFIDENCE_THRESHOLD = 0.3



tracker = initialize_tracker()

print("DeepSORT tracker initialized.")



cap = cv2.VideoCapture("/dev/video4", cv2.CAP_V4L2)

cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)

cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)



if not cap.isOpened():

    print("Cannot open the camera device /dev/video2. Trying default camera (index 0)...")

    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)

    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)



    if not cap.isOpened():

        print("Cannot open default camera (index 0) either. Exiting.")

        exit()

    print("Opened default camera (index 0) instead.")

else:

     print("Successfully opened camera device /dev/video2.")


def generate_frames():
    print("generate_frames function started.")
    try:
        while True:
            print("--- while loop start ---")
            ret, frame = cap.read()
            if not ret:
                print("cap.read() failed: Failed to get frame.")
                print("Failed to grab frame, retrying...")
                print("--- executing continue ---")
                time.sleep(0.1)
                continue

            print("cap.read() successful: Starting frame processing.")

            h, w, _ = frame.shape

            print(f"Frame shape before blob: {frame.shape}, dtype: {frame.dtype}")


            # --- Object Detection (using SSD-MobileNet) ---
            print("--- right before blobFromImage call ---")
            blob = cv2.dnn.blobFromImage(frame, size=(300, 300), swapRB=True, crop=False)
            print("--- right after blobFromImage call ---")

            net.setInput(blob)
            detections = net.forward()
            print(f"--- right after net.forward(), detections shape: {detections.shape} ---")


            current_frame_detections = []
            current_frame_has_target = False

            detections = detections[0, 0]

            # --- 객체 탐지 결과 필터링 루프 (디버깅 print 추가) ---
            print(f"--- Starting detection filtering loop. Number of raw detections: {len(detections)} ---")

            # detection 변수의 실제 내용을 확인하기 위해 루프 안에 print 추가
            for detection in detections:
                # 각 detection의 raw 값 출력 (shape (7,) 형태)
                print(f"Processing detection: {detection}")

                confidence = detection[2]
                class_id = int(detection[1])

                print(f"  Confidence: {confidence:.2f}, Class ID: {class_id}")

                if confidence > CONFIDENCE_THRESHOLD and class_id < len(CLASSES):
                    print(f"  Detection passed confidence/class filter ({confidence:.2f} > {CONFIDENCE_THRESHOLD}, {class_id} < {len(CLASSES)}). Calculating bbox...")
                    # 여기서 오류가 발생할 가능성이 높습니다.
                    # 이 print가 나오는지 확인하여 좌표 계산 직전까지 도달했는지 봅니다.
                    print(f"  --- right before bbox calculation (using w={w}, h={h}) ---")
                    x1 = int(detection[3] * w)
                    y1 = int(detection[4] * h)
                    x2 = int(detection[5] * w)
                    y2 = int(detection[6] * h)
                    # 이 print가 나오는지 확인하여 좌표 계산이 성공했는지 봅니다.
                    print(f"  --- right after bbox calculation ({x1}, {y1}, {x2}, {y2}) ---")


                    print(f"  Calculated bbox: ({x1}, {y1}, {x2}, {y2})")

                    if x2 > x1 and y2 > y1:
                        print(f"  Bbox is valid ({x1}, {y1}, {x2}, {y2}). Appending detection.")
                        current_frame_detections.append([x1, y1, x2, y2, confidence, class_id])

                        if class_id in TARGET_CLASS_IDS:
                            current_frame_has_target = True
                            print("  Target class detected in this frame.")
                    else:
                         print(f"  Bbox is invalid ({x1}, {y1}, {x2}, {y2}). Skipping.")
                # else:
                     # 신뢰도/클래스 필터에 걸린 탐지 결과는 이 print를 출력 (옵션)
                     # print(f"  Detection failed confidence/class filter (confidence={confidence:.2f}, class_id={class_id}). Skipping.")

            print(f"--- Finished detection filtering loop. Filtered detections count: {len(current_frame_detections)} ---")
            # --- 객체 탐지 결과 필터링 루프 끝 ---


            # --- DeepSORT 추적 업데이트 ---
            print(f"Frame {int(time.time())}: DeepSORT input object count = {len(current_frame_detections)}")
            if current_frame_detections:
                 detections_np = np.array(current_frame_detections)

                 bbox_xyxy = detections_np[:, :4]
                 confidences = detections_np[:, 4]
                 classes = detections_np[:, 5]

                 tracked_objects = tracker.update(bbox_xyxy, confidences, classes, frame)

            else:
                 tracked_objects = tracker.update(np.empty((0, 4)), np.empty(0), np.empty(0), frame)

            print(f"Frame {int(time.time())}: DeepSORT tracked object count = {len(tracked_objects)}")
            # --- DeepSORT 추적 업데이트 끝 ---

            # --- LED 제어 ---
            if current_frame_has_target:
                GPIO.output(LED_pin, GPIO.HIGH)
            else:
                GPIO.output(LED_pin, GPIO.LOW)
            # --- LED 제어 끝 ---

            # --- 추적 결과 시각화 ---
            annotated_frame = frame.copy()

            for track_info in tracked_objects:
                 if isinstance(track_info, (list, np.ndarray)) and len(track_info) == 6:
                    x1, y1, x2, y2 = map(int, track_info[:4])
                    track_id = int(track_info[4])
                    class_id_from_track = int(track_info[5])

                    is_valid_bbox = x2 > x1 and y2 > y1

                    print(f"Track ID {track_id}: Raw Coords = ({track_info[0]:.2f}, {track_info[1]:.2f}, {track_info[2]:.2f}, {track_info[3]:.2f})")
                    print(f"Track ID {track_id}: Int Coords = ({x1}, {y1}, {x2}, {y2}), Valid = {is_valid_bbox}")

                    if not is_valid_bbox:
                        print(f"Track ID {track_id}: Skipping drawing due to invalid bbox (x2<x1 or y2<y1)")
                        continue

                    print(f"Track ID {track_id}: Drawing bbox and label at ({x1}, {y1}) to ({x2}, {y2})")

                    class_name = "Unknown"
                    if 0 <= class_id_from_track < len(CLASSES):
                         class_name = CLASSES[class_id_from_track]
                    else:
                         class_name = f"Class {class_id_from_track}"

                    color = (0, 255, 0)

                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

                    label = f"{class_name} ID: {track_id}"

                    (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    text_x = x1
                    text_y = y1 - baseline
                    if text_y < text_height:
                        text_y = y1 + text_height

                    cv2.rectangle(annotated_frame, (text_x, text_y - text_height), (text_x + text_width, text_y), color, -1)
                    cv2.putText(annotated_frame, label, (text_x, text_y - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

            # --- 추적 결과 시각화 끝 ---

            # --- 프레임 인코딩 및 전송 ---
            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            if not ret:
                print("Failed to encode frame.")
                continue

            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            # --- 프레임 인코딩 및 전송 끝 ---

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

        app.run(host='0.0.0.0', port=5000, debug=False)



    except KeyboardInterrupt:

        print("\nServer stopped by user (KeyboardInterrupt).")



    except Exception as e:

        print(f"\nAn error occurred: {e}")



    finally:

        print("Cleaning up resources...")

        if 'cap' in locals() and cap.isOpened():

             cap.release()

             print("Camera released.")

        elif 'cap' in locals():

             print("Camera was not successfully opened.")



        GPIO.cleanup()

        print("GPIO cleaned up.")



        print("Script finished.")

