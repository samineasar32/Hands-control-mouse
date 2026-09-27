"""
Real-time Hand & Body (Pose) Detection using webcam, with extra
"practical tools" modes built on top of it:

    1 - Virtual Mouse       : move the OS cursor with your index finger,
                              pinch thumb+index together to click
    2 - Volume Control      : thumb-index distance sets system volume
    3 - Brightness Control  : thumb-index distance sets screen brightness
    4 - Push-up Counter     : counts push-up reps from your elbow angle
    5 - Squat Counter       : counts squat reps from your knee angle
    6 - Posture Checker     : warns you when your neck/back start to slouch
    0 - Back to plain hand/body detection (no extra mode active)

Base detection uses:
    - OpenCV: camera capture + display
    - MediaPipe Tasks API (HandLandmarker + PoseLandmarker)

Extra modes need a few more packages (see SETUP INSTRUCTIONS at the
bottom of this file). If a package isn't installed, that mode will
simply tell you what's missing instead of crashing the whole script.

Run:
    python hand_body_detection.py

Controls:
    q         - quit
    h         - toggle hand landmark detection on/off
    b         - toggle body (pose) landmark detection on/off
    0-6       - switch active "tool" mode (see list above)
    c         - (mouse mode) recalibrate the pinch-click distance
"""

import os
import time
import math
import urllib.request

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# -----------------------------------------------------------------------
# Optional third-party packages for the extra tool modes.
# Each is imported defensively so the base program still runs even if
# one of these isn't installed - that mode just gets disabled.
# -----------------------------------------------------------------------
try:
    import pyautogui
    pyautogui.FAILSAFE = False  # we manage screen edges ourselves
    MOUSE_AVAILABLE = True
except ImportError:
    MOUSE_AVAILABLE = False

try:
    from ctypes import cast, POINTER
    from comtypes import CLSCTX_ALL
    from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
    VOLUME_AVAILABLE = True
except ImportError:
    VOLUME_AVAILABLE = False

try:
    import screen_brightness_control as sbc
    BRIGHTNESS_AVAILABLE = True
except ImportError:
    BRIGHTNESS_AVAILABLE = False


# -----------------------------------------------------------------------
# Model files (downloaded automatically on first run)
# -----------------------------------------------------------------------
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
HAND_MODEL_PATH = os.path.join(MODEL_DIR, "hand_landmarker.task")
POSE_MODEL_PATH = os.path.join(MODEL_DIR, "pose_landmarker_lite.task")

HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)


def ensure_model(path, url, label):
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print(f"Downloading {label} model (one-time download)...")
        urllib.request.urlretrieve(url, path)
        print(f"{label} model saved to {path}")


# -----------------------------------------------------------------------
# Landmark connection maps (the new Tasks API no longer ships the old
# mp.solutions.drawing_utils helpers, so we draw manually)
# -----------------------------------------------------------------------
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]

POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (29, 31), (27, 31),
    (24, 26), (26, 28), (28, 30), (30, 32), (28, 32),
]

# Named indices for the pose landmarks we use in the tool modes
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_ELBOW, RIGHT_ELBOW = 13, 14
LEFT_WRIST, RIGHT_WRIST = 15, 16
LEFT_HIP, RIGHT_HIP = 23, 24
LEFT_KNEE, RIGHT_KNEE = 25, 26
LEFT_ANKLE, RIGHT_ANKLE = 27, 28
LEFT_EAR, RIGHT_EAR = 7, 8


def draw_landmarks(frame, landmarks, connections, point_color, line_color):
    h, w = frame.shape[:2]
    points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for start_idx, end_idx in connections:
        if start_idx < len(points) and end_idx < len(points):
            cv2.line(frame, points[start_idx], points[end_idx], line_color, 2)
    for x, y in points:
        cv2.circle(frame, (x, y), 4, point_color, -1)


def pixel_distance(p1, p2, w, h):
    return math.hypot((p1.x - p2.x) * w, (p1.y - p2.y) * h)


def calculate_angle(a, b, c):
    """Angle at point b (in degrees), given three landmarks a-b-c."""
    ang = math.degrees(
        math.atan2(c.y - b.y, c.x - b.x) - math.atan2(a.y - b.y, a.x - b.x)
    )
    ang = abs(ang)
    if ang > 180:
        ang = 360 - ang
    return ang


def tilt_from_vertical(top, bottom):
    """Angle (degrees) between the top->bottom vector and straight-down vertical."""
    dx = bottom.x - top.x
    dy = bottom.y - top.y
    angle = math.degrees(math.atan2(abs(dx), abs(dy)))
    return angle


# -----------------------------------------------------------------------
# Rep counter helper (shared logic for push-ups and squats)
# -----------------------------------------------------------------------
class RepCounter:
    def __init__(self, down_angle, up_angle):
        self.down_angle = down_angle
        self.up_angle = up_angle
        self.stage = "up"
        self.count = 0

    def update(self, angle):
        if angle < self.down_angle:
            self.stage = "down"
        elif angle > self.up_angle and self.stage == "down":
            self.stage = "up"
            self.count += 1
        return self.stage, self.count


def main():
    ensure_model(HAND_MODEL_PATH, HAND_MODEL_URL, "hand landmarker")
    ensure_model(POSE_MODEL_PATH, POSE_MODEL_URL, "pose landmarker")

    BaseOptions = mp_python.BaseOptions
    VisionRunningMode = mp_vision.RunningMode

    hand_options = mp_vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=HAND_MODEL_PATH),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    pose_options = mp_vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=POSE_MODEL_PATH),
        running_mode=VisionRunningMode.VIDEO,
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    hand_landmarker = mp_vision.HandLandmarker.create_from_options(hand_options)
    pose_landmarker = mp_vision.PoseLandmarker.create_from_options(pose_options)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam. Check your camera connection/permissions.")
        return

    detect_hands = True
    detect_body = True
    prev_time = 0
    start_time = time.time()

    # mode: 0 = plain detection, 1 = mouse, 2 = volume, 3 = brightness,
    #       4 = push-ups, 5 = squats, 6 = posture
    mode = 0
    mode_names = {
        0: "Plain Detection",
        1: "Virtual Mouse",
        2: "Volume Control",
        3: "Brightness Control",
        4: "Push-up Counter",
        5: "Squat Counter",
        6: "Posture Checker",
    }

    # --- Virtual mouse state ---
    screen_w, screen_h = pyautogui.size() if MOUSE_AVAILABLE else (1920, 1080)
    smoothing = 5.0
    prev_mouse_x, prev_mouse_y = screen_w / 2, screen_h / 2
    click_distance_threshold = 35  # pixels; press 'c' to recalibrate
    is_clicking = False
    frame_margin = 100  # inset region mapped to the full screen

    # --- Volume control state (Windows only, via pycaw) ---
    volume_interface = None
    vol_min, vol_max = -65.0, 0.0
    if VOLUME_AVAILABLE:
        try:
            speakers = AudioUtilities.GetSpeakers()
            interface = speakers.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
            volume_interface = cast(interface, POINTER(IAudioEndpointVolume))
            vol_min, vol_max = volume_interface.GetVolumeRange()[:2]
        except Exception as e:
            print(f"Volume control unavailable: {e}")
            volume_interface = None

    # --- Rep counters ---
    pushup_counter = RepCounter(down_angle=90, up_angle=160)
    squat_counter = RepCounter(down_angle=100, up_angle=165)

    print("Starting camera feed.")
    print("Keys: q=quit | h=toggle hands | b=toggle body | 0-6=switch mode | c=recalibrate click")
    if not MOUSE_AVAILABLE:
        print("NOTE: install 'pyautogui' to enable Virtual Mouse mode.")
    if not VOLUME_AVAILABLE:
        print("NOTE: install 'pycaw' and 'comtypes' (Windows only) to enable Volume Control mode.")
    if not BRIGHTNESS_AVAILABLE:
        print("NOTE: install 'screen-brightness-control' to enable Brightness Control mode.")

    while True:
        success, frame = cap.read()
        if not success:
            print("Error: Failed to read frame from webcam.")
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = int((time.time() - start_time) * 1000)

        pose_result = None
        if detect_body:
            pose_result = pose_landmarker.detect_for_video(mp_image, timestamp_ms)
            if pose_result.pose_landmarks:
                for pose_landmarks in pose_result.pose_landmarks:
                    draw_landmarks(
                        frame, pose_landmarks, POSE_CONNECTIONS,
                        point_color=(245, 117, 66), line_color=(245, 66, 230),
                    )

        hand_result = None
        if detect_hands:
            hand_result = hand_landmarker.detect_for_video(mp_image, timestamp_ms)
            if hand_result.hand_landmarks:
                for hand_landmarks, handedness in zip(
                    hand_result.hand_landmarks, hand_result.handedness
                ):
                    draw_landmarks(
                        frame, hand_landmarks, HAND_CONNECTIONS,
                        point_color=(0, 255, 255), line_color=(0, 200, 0),
                    )
                    wrist = hand_landmarks[0]
                    cx, cy = int(wrist.x * w), int(wrist.y * h)
                    raw_label = handedness[0].category_name
                    label = "Right" if raw_label == "Left" else "Left"
                    cv2.putText(
                        frame, label, (cx - 20, cy - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA,
                    )

        # -----------------------------------------------------------
        # MODE 1: Virtual Mouse
        # -----------------------------------------------------------
        if mode == 1:
            if not MOUSE_AVAILABLE:
                cv2.putText(frame, "pyautogui not installed", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            elif hand_result and hand_result.hand_landmarks:
                lm = hand_result.hand_landmarks[0]
                index_tip = lm[8]
                thumb_tip = lm[4]

                # map the active region (inset from frame edges) to full screen
                ix = min(max(index_tip.x * w, frame_margin), w - frame_margin)
                iy = min(max(index_tip.y * h, frame_margin), h - frame_margin)
                screen_x = ((ix - frame_margin) / (w - 2 * frame_margin)) * screen_w
                screen_y = ((iy - frame_margin) / (h - 2 * frame_margin)) * screen_h

                # smooth the movement so the cursor doesn't jitter
                curr_x = prev_mouse_x + (screen_x - prev_mouse_x) / smoothing
                curr_y = prev_mouse_y + (screen_y - prev_mouse_y) / smoothing
                pyautogui.moveTo(curr_x, curr_y)
                prev_mouse_x, prev_mouse_y = curr_x, curr_y

                dist = pixel_distance(thumb_tip, index_tip, w, h)
                if dist < click_distance_threshold and not is_clicking:
                    pyautogui.click()
                    is_clicking = True
                elif dist >= click_distance_threshold:
                    is_clicking = False

                cv2.putText(frame, f"Pinch dist: {int(dist)} (threshold {click_distance_threshold})",
                            (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                cv2.rectangle(frame, (frame_margin, frame_margin),
                              (w - frame_margin, h - frame_margin), (255, 0, 0), 1)
            else:
                cv2.putText(frame, "Show one hand to control the mouse", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

        # -----------------------------------------------------------
        # MODE 2: Volume Control
        # -----------------------------------------------------------
        elif mode == 2:
            if not VOLUME_AVAILABLE or volume_interface is None:
                cv2.putText(frame, "pycaw/comtypes not available (Windows only)", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            elif hand_result and hand_result.hand_landmarks:
                lm = hand_result.hand_landmarks[0]
                thumb_tip, index_tip = lm[4], lm[8]
                dist = pixel_distance(thumb_tip, index_tip, w, h)
                dist = max(20, min(dist, 200))
                vol_percent = (dist - 20) / (200 - 20)
                volume_interface.SetMasterVolumeLevel(
                    vol_min + vol_percent * (vol_max - vol_min), None
                )
                bar_h = int(200 * vol_percent)
                cv2.rectangle(frame, (30, 250 - bar_h), (60, 250), (0, 255, 0), -1)
                cv2.rectangle(frame, (30, 50), (60, 250), (255, 255, 255), 2)
                cv2.putText(frame, f"Vol: {int(vol_percent * 100)}%", (20, 270),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            else:
                cv2.putText(frame, "Pinch thumb+index on one hand to set volume", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

        # -----------------------------------------------------------
        # MODE 3: Brightness Control
        # -----------------------------------------------------------
        elif mode == 3:
            if not BRIGHTNESS_AVAILABLE:
                cv2.putText(frame, "screen_brightness_control not installed", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            elif hand_result and hand_result.hand_landmarks:
                lm = hand_result.hand_landmarks[0]
                thumb_tip, index_tip = lm[4], lm[8]
                dist = pixel_distance(thumb_tip, index_tip, w, h)
                dist = max(20, min(dist, 200))
                brightness_percent = int(((dist - 20) / (200 - 20)) * 100)
                try:
                    sbc.set_brightness(brightness_percent)
                except Exception as e:
                    cv2.putText(frame, f"Brightness error: {e}", (10, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                bar_h = int(200 * (brightness_percent / 100))
                cv2.rectangle(frame, (30, 250 - bar_h), (60, 250), (0, 255, 255), -1)
                cv2.rectangle(frame, (30, 50), (60, 250), (255, 255, 255), 2)
                cv2.putText(frame, f"Brightness: {brightness_percent}%", (20, 270),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            else:
                cv2.putText(frame, "Pinch thumb+index on one hand to set brightness", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

        # -----------------------------------------------------------
        # MODE 4: Push-up Counter
        # -----------------------------------------------------------
        elif mode == 4:
            if pose_result and pose_result.pose_landmarks:
                lm = pose_result.pose_landmarks[0]
                angle = calculate_angle(lm[RIGHT_SHOULDER], lm[RIGHT_ELBOW], lm[RIGHT_WRIST])
                stage, count = pushup_counter.update(angle)
                ex, ey = int(lm[RIGHT_ELBOW].x * w), int(lm[RIGHT_ELBOW].y * h)
                cv2.putText(frame, f"{int(angle)} deg", (ex + 10, ey),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                cv2.putText(frame, f"Push-ups: {count}  ({stage})", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Step back so your full arm is visible (side-on works best)",
                            (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

        # -----------------------------------------------------------
        # MODE 5: Squat Counter
        # -----------------------------------------------------------
        elif mode == 5:
            if pose_result and pose_result.pose_landmarks:
                lm = pose_result.pose_landmarks[0]
                angle = calculate_angle(lm[RIGHT_HIP], lm[RIGHT_KNEE], lm[RIGHT_ANKLE])
                stage, count = squat_counter.update(angle)
                kx, ky = int(lm[RIGHT_KNEE].x * w), int(lm[RIGHT_KNEE].y * h)
                cv2.putText(frame, f"{int(angle)} deg", (kx + 10, ky),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                cv2.putText(frame, f"Squats: {count}  ({stage})", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Step back so your full leg is visible (side-on works best)",
                            (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

        # -----------------------------------------------------------
        # MODE 6: Posture Checker
        # -----------------------------------------------------------
        elif mode == 6:
            if pose_result and pose_result.pose_landmarks:
                lm = pose_result.pose_landmarks[0]
                angle = tilt_from_vertical(lm[RIGHT_SHOULDER], lm[RIGHT_EAR])
                slouching = angle > 30
                color = (0, 0, 255) if slouching else (0, 255, 0)
                msg = "SLOUCHING - straighten up!" if slouching else "Posture looks good"
                cv2.putText(frame, f"{msg} ({int(angle)} deg)", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            else:
                cv2.putText(frame, "Sit/stand side-on to the camera for posture checks",
                            (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

        # -----------------------------------------------------------
        # HUD: FPS + status + mode name
        # -----------------------------------------------------------
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time else 0
        prev_time = curr_time
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Mode: {mode_names[mode]} (0-6 to switch)",
                    (w - 430, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        status = f"Hands: {'ON' if detect_hands else 'OFF'} (h) | Body: {'ON' if detect_body else 'OFF'} (b) | Quit: q"
        cv2.putText(frame, status, (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow("Hand & Body Detection", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('h'):
            detect_hands = not detect_hands
        elif key == ord('b'):
            detect_body = not detect_body
        elif key == ord('c'):
            click_distance_threshold = 35  # reset calibration
        elif key in (ord('0'), ord('1'), ord('2'), ord('3'), ord('4'), ord('5'), ord('6')):
            mode = int(chr(key))
            # make sure the detector this mode needs is switched on
            if mode in (1, 2, 3):
                detect_hands = True
            elif mode in (4, 5, 6):
                detect_body = True

    cap.release()
    cv2.destroyAllWindows()
    hand_landmarker.close()
    pose_landmarker.close()


if __name__ == "__main__":
    main()

# -----------------------------------------------------------------------
# SETUP INSTRUCTIONS
# -----------------------------------------------------------------------
# Base requirements (always needed):
#     pip install opencv-python mediapipe
#
# For Virtual Mouse mode (mode 1):
#     pip install pyautogui
#
# For Volume Control mode (mode 2) - Windows only:
#     pip install pycaw comtypes
#
# For Brightness Control mode (mode 3):
#     pip install screen-brightness-control
#   (On Windows this works out of the box for most laptop/monitor
#    setups. On Linux it may require xrandr or a compatible backend.)
#
# Push-up Counter, Squat Counter, and Posture Checker (modes 4-6) only
# need the base requirements - no extra packages.
#
# Quick install-everything command:
#     pip install opencv-python mediapipe pyautogui pycaw comtypes screen-brightness-control
#
# Tips:
# - For the push-up/squat counters, stand or get into position side-on
#   to the camera so your arm/leg angle is clearly visible.
# - For Virtual Mouse and Volume/Brightness modes, only show ONE hand at
#   a time to avoid ambiguous readings.
# - Press 'c' at any time to reset the pinch-click sensitivity if
#   clicking feels too sensitive or not sensitive enough.
# -----------------------------------------------------------------------