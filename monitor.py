import cv2
import mediapipe as mp
from ultralytics import YOLO
import pygame
import math
import os
import json
import time

# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "models", "yolov8n.pt")
ALARM_PATH = os.path.join(BASE_DIR, "alarm", "alarm.mp3")
COUNT_FILE = os.path.join(BASE_DIR, "monitor_counts.json")

print("BASE DIRECTORY :", BASE_DIR)
print("MODEL          :", MODEL_PATH)
print("ALARM          :", ALARM_PATH)

# ============================================================
# PERSISTENT COUNTS
# ============================================================

def load_counts():

    if os.path.exists(COUNT_FILE):

        try:
            with open(COUNT_FILE, "r") as f:
                data = json.load(f)

            return (
                data.get("drowsiness", 0),
                data.get("yawn", 0),
                data.get("head_pose", 0),
                data.get("phone", 0)
            )

        except:
            pass

    return 0, 0, 0, 0


def save_counts():

    data = {
        "drowsiness": drowsiness_count,
        "yawn": yawn_count,
        "head_pose": head_pose_count,
        "phone": phone_count
    }

    with open(COUNT_FILE, "w") as f:
        json.dump(data, f, indent=4)


drowsiness_count, yawn_count, head_pose_count, phone_count = load_counts()

print()
print("================================")
print("PREVIOUS COUNTS")
print("Drowsiness :", drowsiness_count)
print("Yawn       :", yawn_count)
print("Head Pose  :", head_pose_count)
print("Phone      :", phone_count)
print("================================")

# ============================================================
# MEDIAPIPE
# ============================================================

mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ============================================================
# YOLO
# ============================================================

model = YOLO(MODEL_PATH)

# ============================================================
# ALARM
# ============================================================

pygame.mixer.init()

alarm_playing = False


def play_alarm():

    global alarm_playing

    if not alarm_playing:

        if not os.path.exists(ALARM_PATH):

            print("ERROR: Alarm file not found:")
            print(ALARM_PATH)
            return

        try:

            pygame.mixer.music.load(ALARM_PATH)
            pygame.mixer.music.play(-1)

            alarm_playing = True

            print("ALARM ON")

        except Exception as e:

            print("Alarm Error:", e)


def stop_alarm():

    global alarm_playing

    if alarm_playing:

        pygame.mixer.music.stop()

        alarm_playing = False

        print("ALARM OFF")


# ============================================================
# DISTANCE
# ============================================================

def distance(p1, p2):

    return math.sqrt(
        (p1.x - p2.x) ** 2 +
        (p1.y - p2.y) ** 2
    )


# ============================================================
# EAR - EYE ASPECT RATIO
# ============================================================

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]


def eye_aspect_ratio(landmarks, eye):

    vertical1 = distance(
        landmarks[eye[1]],
        landmarks[eye[5]]
    )

    vertical2 = distance(
        landmarks[eye[2]],
        landmarks[eye[4]]
    )

    horizontal = distance(
        landmarks[eye[0]],
        landmarks[eye[3]]
    )

    if horizontal == 0:
        return 0

    return (vertical1 + vertical2) / (2 * horizontal)


# ============================================================
# MOUTH
# ============================================================

UPPER_LIP = 13
LOWER_LIP = 14

YAWN_THRESHOLD = 0.025


# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():

    print("ERROR: Camera not opening")
    exit()


# ============================================================
# DETECTION SETTINGS
# ============================================================

EAR_THRESHOLD = 0.23

CLOSED_FRAME_LIMIT = 20

closed_frames = 0
yawn_frames = 0

# Prevent counting same event repeatedly
drowsiness_active = False
yawn_active = False
head_active = False
phone_active = False


# ============================================================
# HEAD POSE
# ============================================================

previous_head_direction = "CENTER"


# ============================================================
# PHONE
# ============================================================

# COCO class ID for cell phone
PHONE_CLASS_ID = 67


print()
print("======================================")
print("SMART DRIVER MONITORING STARTED")
print("Press Q to stop")
print("======================================")


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:

        print("Camera frame error")
        break

    frame = cv2.flip(frame, 1)

    height, width, _ = frame.shape

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    # ========================================================
    # FACE DETECTION
    # ========================================================

    results = face_mesh.process(rgb)

    drowsiness_detected = False
    yawn_detected = False
    head_detected = False

    # ========================================================
    # FACE FEATURES
    # ========================================================

    if results.multi_face_landmarks:

        face = results.multi_face_landmarks[0]

        landmarks = face.landmark

        # ----------------------------------------------------
        # DROWSINESS
        # ----------------------------------------------------

        left_ear = eye_aspect_ratio(
            landmarks,
            LEFT_EYE
        )

        right_ear = eye_aspect_ratio(
            landmarks,
            RIGHT_EYE
        )

        avg_ear = (left_ear + right_ear) / 2

        cv2.putText(
            frame,
            f"EAR: {avg_ear:.2f}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        if avg_ear < EAR_THRESHOLD:

            closed_frames += 1

        else:

            closed_frames = 0
            drowsiness_active = False

        if closed_frames >= CLOSED_FRAME_LIMIT:

            drowsiness_detected = True

            cv2.putText(
                frame,
                "DROWSINESS DETECTED",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                3
            )

            # Count only once per drowsiness episode
            if not drowsiness_active:

                drowsiness_count += 1

                save_counts()

                print(
                    "DROWSINESS COUNT =",
                    drowsiness_count
                )

                drowsiness_active = True

        # ----------------------------------------------------
        # YAWN
        # ----------------------------------------------------

        mouth_open = distance(
            landmarks[UPPER_LIP],
            landmarks[LOWER_LIP]
        )

        if mouth_open > YAWN_THRESHOLD:

            yawn_frames += 1

        else:

            yawn_frames = 0
            yawn_active = False

        if yawn_frames >= 10:

            yawn_detected = True

            cv2.putText(
                frame,
                "YAWN DETECTED",
                (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 255),
                3
            )

            # Count once per yawn
            if not yawn_active:

                yawn_count += 1

                save_counts()

                print(
                    "YAWN COUNT =",
                    yawn_count
                )

                yawn_active = True

        # ----------------------------------------------------
        # HEAD POSE
        # ----------------------------------------------------

        nose = landmarks[1]

        nose_x = nose.x * width

        if nose_x < width / 3:

            head_direction = "LEFT"

        elif nose_x > (2 * width / 3):

            head_direction = "RIGHT"

        else:

            head_direction = "CENTER"

        cv2.putText(
            frame,
            "HEAD: " + head_direction,
            (20, 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        # Count only when direction changes
        if head_direction != "CENTER":

            head_detected = True

            if not head_active:

                head_pose_count += 1

                save_counts()

                print(
                    "HEAD POSE COUNT =",
                    head_pose_count
                )

                head_active = True

        else:

            head_active = False

        previous_head_direction = head_direction

    # ========================================================
    # PHONE DETECTION - YOLO
    # ========================================================

    phone_detected = False

    yolo_results = model(
        frame,
        verbose=False,
        conf=0.50
    )

    for result in yolo_results:

        if result.boxes is None:
            continue

        for box in result.boxes:

            class_id = int(box.cls[0])

            confidence = float(box.conf[0])

            if class_id == PHONE_CLASS_ID and confidence >= 0.50:

                phone_detected = True

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0]
                )

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 0, 255),
                    2
                )

                cv2.putText(
                    frame,
                    "PHONE",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2
                )

    if phone_detected:

        cv2.putText(
            frame,
            "PHONE USAGE DETECTED",
            (20, 190),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            3
        )

        if not phone_active:

            phone_count += 1

            save_counts()

            print(
                "PHONE COUNT =",
                phone_count
            )

            phone_active = True

    else:

        phone_active = False

    # ========================================================
    # ALARM
    # ========================================================

    if (
        drowsiness_detected
        or yawn_detected
        or phone_detected
    ):

        play_alarm()

    else:

        stop_alarm()

    # ========================================================
    # DISPLAY COUNTERS
    # ========================================================

    cv2.putText(
        frame,
        f"Drowsiness: {drowsiness_count}",
        (width - 280, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Yawn: {yawn_count}",
        (width - 280, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Head Pose: {head_pose_count}",
        (width - 280, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Phone: {phone_count}",
        (width - 280, 130),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    # ========================================================
    # SHOW CAMERA
    # ========================================================

    cv2.imshow(
        "Smart Driver Management System",
        frame
    )

    # Q TO EXIT
    if cv2.waitKey(1) & 0xFF == ord("q"):

        break


# ============================================================
# STOP
# ============================================================

stop_alarm()

cap.release()

cv2.destroyAllWindows()

pygame.quit()

save_counts()

print()
print("======================================")
print("MONITORING STOPPED")
print("======================================")
print("Drowsiness :", drowsiness_count)
print("Yawn       :", yawn_count)
print("Head Pose  :", head_pose_count)
print("Phone      :", phone_count)
print("======================================")