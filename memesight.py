import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import threading
import queue
import time
from PIL import Image 

# ==========================================
#             FILE PATHS (ALL MEMES)
# ==========================================

PATH_TONGUE_SHAKE      = r"memes\freaky-orca.gif"
PATH_HAND_ON_CHIN      = r"memes\monkeythink.jpg"
PATH_HEAD_NOD          = r"memes\ishowspeed.gif"
PATH_ONE_FINGER_UP     = r"memes\monkeyrealize.jpeg"
PATH_HANDS_ON_HEAD     = r"memes\ishowspeed-wow.gif"
PATH_NAMASTE           = r"memes\freaky-sonic.mp4"
PATH_THUMBS_UP         = r"memes\humbsupmonkey.png"
PATH_IDLE_STARE        = r"memes\monkeytruth.jpg"
IDLE_MIN_FRAMES = 70  # ~1+ second of stillness required
idle_hold = 0         # internal counter
palm_dist_buffer = deque(maxlen=20)

# ==========================================
#          GIF LOADING ENGINE (OPTION A)
# ==========================================

def load_gif_frames(path):
    """Load all GIF frames as a list of BGR numpy arrays."""
    frames = []
    try:
        gif = Image.open(path)
    except:
        print(f"❌ Error loading GIF: {path}")
        return frames

    try:
        while True:
            frame = gif.convert("RGB")
            np_frame = np.array(frame)
            bgr = cv2.cvtColor(np_frame, cv2.COLOR_RGB2BGR)
            frames.append(bgr)
            gif.seek(gif.tell() + 1)
    except EOFError:
        pass

    return frames


# Preload all GIFs at start for speed
GIF_TONGUE_SHAKE  = load_gif_frames(PATH_TONGUE_SHAKE)
GIF_HEAD_NOD      = load_gif_frames(PATH_HEAD_NOD)
GIF_HANDS_ON_HEAD = load_gif_frames(PATH_HANDS_ON_HEAD)


# ==========================================
#       IMAGE LOADING FOR STILL MEMES
# ==========================================

IMG_HAND_ON_CHIN  = cv2.imread(PATH_HAND_ON_CHIN)
IMG_ONE_FINGER_UP = cv2.imread(PATH_ONE_FINGER_UP)
IMG_IDLE_STARE    = cv2.imread(PATH_IDLE_STARE)
IMG_THUMBS_UP = cv2.imread(PATH_THUMBS_UP)
if IMG_THUMBS_UP is None:
    print("❌ ERROR: Cannot load thumbs up image")

if IMG_HAND_ON_CHIN is None:
    print("❌ ERROR: Cannot load monkeythink.jpg")

if IMG_ONE_FINGER_UP is None:
    print("❌ ERROR: Cannot load monkeyrealize.jpeg")

if IMG_IDLE_STARE is None:
    print("❌ ERROR: Cannot load monkeytruth.jpg")


# ==========================================
#            VIDEO PLAYBACK THREAD
# ==========================================

video_queue = queue.Queue(maxsize=1)
play_video_flag = threading.Event()
video_done_flag = threading.Event()

CURRENT_VIDEO = None  # path of currently playing video


def video_player_thread():
    """Thread that plays MP4 meme videos ONCE and outputs frames to queue."""
    global CURRENT_VIDEO

    while True:
        play_video_flag.wait()

        vid = cv2.VideoCapture(CURRENT_VIDEO)

        if not vid.isOpened():
            print("❌ Unable to open video:", CURRENT_VIDEO)
            play_video_flag.clear()
            continue

        while play_video_flag.is_set():
            ok, frame = vid.read()
            if not ok:
                break  # stop at end of video

            if not video_queue.full():
                video_queue.put(frame)

            time.sleep(0.01)

        vid.release()
        video_done_flag.set()
        play_video_flag.clear()


threading.Thread(target=video_player_thread, daemon=True).start()


# ==========================================
#      STATE MANAGEMENT + GLOBAL SETTINGS
# ==========================================

# Gesture cooldown (prevents spamming)
COOLDOWN_FRAMES = 30
cooldown = 0

# For sustained gestures
REQUIRED_FRAMES = 6

# Buffers for nod & shake detection
nose_x_buffer = deque(maxlen=15)
nose_y_buffer = deque(maxlen=20)

# Flags for which meme is active
active_mode = "idle"     # idle / gif / video / image
current_gif_frames = []
current_gif_index = 0

# ==========================================
#         MEDIAPIPE INITIALIZATION
# ==========================================

mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(max_num_faces=1, refine_landmarks=True)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)


# ==========================================
#       GESTURE DETECTION FUNCTIONS
# ==========================================

# ------------------------------
# 1. Tongue OUT detection
# ------------------------------
def detect_tongue(lm):
    upper = lm[13].y
    lower = lm[14].y
    tongue = lm[16].y
    return (lower - upper) > 0.01 and (tongue - lower) > 0.01


# ------------------------------
# 2. Head SHAKE detection
# ------------------------------
def detect_head_shake():
    if len(nose_x_buffer) < 15:
        return False
    motion = max(nose_x_buffer) - min(nose_x_buffer)
    return motion > 0.03


# ------------------------------
# 3. Head NOD detection
# ------------------------------
def detect_head_nod():
    if len(nose_y_buffer) < 15:
        return False

    y_vals = list(nose_y_buffer)
    min_y = min(y_vals)
    max_y = max(y_vals)

    # Must move DOWN then UP or UP then DOWN
    motion = max_y - min_y

    return motion > 0.07



# ------------------------------
# 4. Hand ON CHIN gesture
# ------------------------------
def detect_hand_on_chin(face_lm, hand_lm):
    chin = face_lm[199]
    tip = hand_lm.landmark[8]  # index finger tip

    dx = chin.x - tip.x
    dy = chin.y - tip.y

    dist = (dx*dx + dy*dy)**0.5
    return dist < 0.12


# ------------------------------
# 5. One finger pointing UP
# ------------------------------
def detect_one_finger_up(hand_lm):
    index_up = hand_lm.landmark[8].y < hand_lm.landmark[7].y
    mid_down = hand_lm.landmark[12].y > hand_lm.landmark[11].y
    ring_down = hand_lm.landmark[16].y > hand_lm.landmark[15].y
    pinky_down = hand_lm.landmark[20].y > hand_lm.landmark[19].y

    return index_up and mid_down and ring_down and pinky_down


# ------------------------------
# 6. Both hands ON HEAD
# ------------------------------
def detect_hands_on_head(face_lm, hand_list):
    if len(hand_list) < 2:
        return False

    forehead_y = face_lm[10].y  # forehead landmark

    tips = [hand.landmark[8].y for hand in hand_list]

    return all(tip < forehead_y + 0.02 for tip in tips)


def detect_rubbing_palms(hand_list):
    # Need 2 hands
    if len(hand_list) < 2:
        return False

    h1 = hand_list[0].landmark
    h2 = hand_list[1].landmark

    # Palm centers (landmark 9)
    p1 = h1[9]
    p2 = h2[9]

    # Distance between palms
    dx = p1.x - p2.x
    dy = p1.y - p2.y
    dist = (dx*dx + dy*dy) ** 0.5

    # Record in buffer
    palm_dist_buffer.append(dist)

    # Need enough data
    if len(palm_dist_buffer) < 20:
        return False

    # Measure oscillation strength
    diff = max(palm_dist_buffer) - min(palm_dist_buffer)

    # Threshold tuning
    rubbing_detected = diff > 0.06  # motion amplitude required

    # Palms must be close enough to count as rubbing
    close_enough = dist < 0.18

    return rubbing_detected and close_enough


# ------------------------------
# 8. Thumbs up gesture
# ------------------------------
'''
def detect_thumbs_up(hand_lm):
    thumb_tip = hand_lm.landmark[4].y
    thumb_ip  = hand_lm.landmark[3].y

    index_tip = hand_lm.landmark[8].y

    return (thumb_tip < thumb_ip) and (index_tip > hand_lm.landmark[5].y)
'''

# ------------------------------
# 9. Idle stare (face stable + eyes forward)
# ------------------------------
def detect_idle_stare(face_lm):
    """
    Detects when the user is truly idle:
      - Head barely moving
      - Face centered
      - Eyes looking straight
    """

    # Must have enough buffer data
    if len(nose_x_buffer) < 15 or len(nose_y_buffer) < 15:
        return False

    # STRONG stability requirement (reduced thresholds)
    stable_x = (max(nose_x_buffer) - min(nose_x_buffer)) < 0.005
    stable_y = (max(nose_y_buffer) - min(nose_y_buffer)) < 0.005

    if not (stable_x and stable_y):
        return False

    # Eyes forward (stricter requirement)
    left_eye = face_lm[468]
    right_eye = face_lm[473]

    facing_forward = abs(left_eye.x - right_eye.x) < 0.08

    return facing_forward

# ==========================================
#   PLAYBACK ENGINE + GESTURE PRIORITY SYSTEM
# ==========================================

def activate_gif(gif_frames):
    """Prepare system to play a GIF animation once."""
    global active_mode, current_gif_frames, current_gif_index

    active_mode = "gif"
    current_gif_frames = gif_frames
    current_gif_index = 0
    video_done_flag.set()       # no video playing
    play_video_flag.clear()     # ensure video thread stops


def activate_image(img):
    """Show a static image."""
    global active_mode, current_gif_frames

    active_mode = "image"
    current_gif_frames = [img]  # stored as 1-frame list
    video_done_flag.set()
    play_video_flag.clear()


def activate_video(path):
    """Trigger video playback."""
    global active_mode, CURRENT_VIDEO

    active_mode = "video"
    CURRENT_VIDEO = path
    video_done_flag.clear()
    play_video_flag.set()       # start video thread


def set_idle_state():
    """Show idle meme and wait for next gesture."""
    global active_mode, current_gif_frames, current_gif_index

    active_mode = "idle"
    current_gif_frames = [IMG_IDLE_STARE]
    current_gif_index = 0
    play_video_flag.clear()
    video_done_flag.set()


# ==========================================
#         PRIORITY ORDER FOR GESTURES
# ==========================================

def evaluate_gesture_priority(
        tongue_shake,
        hand_chin,
        head_nod,
        one_finger_up,
        hands_on_head,
        rubbing_palms_flag,
        idle_stare
    ):
    """
    Determines which meme to trigger based on priority.
    Returns a string key representing the selected gesture.
    """

    if tongue_shake:
        return "tongue_shake"

    if hand_chin:
        return "hand_chin"

    if head_nod:
        return "head_nod"

    if one_finger_up:
        return "one_finger_up"

    if hands_on_head:
        return "hands_on_head"

    if rubbing_palms_flag:
        return "rubbing_palms"

    if idle_stare:
        return "idle"

    return "none"


# ==========================================
#                 MAIN LOOP
# ==========================================

cap = cv2.VideoCapture(0)

frame_counter = 0
gesture_hold = 0   # frames gesture is sustained


while True:
    ok, frame = cap.read()
    if not ok:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    cam_h = 480
    cam_w = 640

    resized_cam = cv2.resize(frame, (cam_w, cam_h))

    # ------------------------------------------
    # Run FaceMesh + Hands
    # ------------------------------------------
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_result = face_mesh.process(rgb)
    hands_result = hands.process(rgb)

    face_lm = None
    hand_lms = []

    if face_result.multi_face_landmarks:
        face_lm = face_result.multi_face_landmarks[0].landmark

        # Update nose movement buffers
        nose_x_buffer.append(face_lm[1].x)
        nose_y_buffer.append(face_lm[1].y)

    if hands_result.multi_hand_landmarks:
        hand_lms = hands_result.multi_hand_landmarks


    # ------------------------------------------
    # Detect all gestures
    # ------------------------------------------
    if face_lm:
        tongue_out = detect_tongue(face_lm)
        shake = detect_head_shake()
    
        nod = detect_head_nod()

        # IDLE stare logic using hold counter
        if detect_idle_stare(face_lm) and len(hand_lms) == 0:
            idle_hold += 1
        else:
            idle_hold = 0

        idle = (idle_hold >= IDLE_MIN_FRAMES)
    else:
        tongue_out = shake = nod = idle = False
        idle_hold = 0  # reset if face not detected
    tongue_shake = tongue_out and shake

    hand_chin = False
    one_finger_up = False
    hands_on_head = False
    rubbing_palms_flag = False

    if hand_lms:
    # 1 hand
        if len(hand_lms) == 1:
            hl = hand_lms[0]
            if face_lm:
                hand_chin = detect_hand_on_chin(face_lm, hl)

            one_finger_up = detect_one_finger_up(hl)
            # thumbs_up = detect_thumbs_up(hl)  # <-- DISABLED

    # 2+ hands
    if len(hand_lms) >= 2 and face_lm:
        hands_on_head = detect_hands_on_head(face_lm, hand_lms)
        rubbing_palms_flag = detect_rubbing_palms(hand_lms)


    # ------------------------------------------
    # Apply PRIORITY engine
    # ------------------------------------------
    highest = evaluate_gesture_priority(
        tongue_shake,
        hand_chin,
        nod,
        one_finger_up,
        hands_on_head,
        rubbing_palms_flag,
        idle
    )

    # ------------------------------------------
    # Apply COOLDOWN (prevents constant triggering)
    # ------------------------------------------
    if cooldown > 0:
        cooldown -= 1
        highest = "none"  # ignore new triggers

    # ------------------------------------------
    # Trigger meme only when gesture is sustained
    # ------------------------------------------
    if highest != "none":
        # Thumbs up allows faster triggering
        if highest == "thumbs_up":
            gesture_hold += 2     # double count
        else:
            gesture_hold += 1
    else:
        gesture_hold = 0


    if gesture_hold >= REQUIRED_FRAMES and cooldown == 0:
        print("🎯 Trigger:", highest)

        # Reset
        gesture_hold = 0
        cooldown = COOLDOWN_FRAMES

        # Trigger meme based on gesture key
        if highest == "tongue_shake":
            activate_gif(GIF_TONGUE_SHAKE)

        elif highest == "hand_chin":
            activate_image(IMG_HAND_ON_CHIN)

        elif highest == "head_nod":
            activate_gif(GIF_HEAD_NOD)

        elif highest == "one_finger_up":
            activate_image(IMG_ONE_FINGER_UP)

        elif highest == "hands_on_head":
            activate_gif(GIF_HANDS_ON_HEAD)

        elif highest == "rubbing_palms":
            activate_video(PATH_NAMASTE)


 #       elif highest == "thumbs_up":
  #          activate_gif(IMG_THUMBS_UP)

        elif highest == "idle":
            set_idle_state()


    # ------------------------------------------
    # BUILD RIGHT PANEL (GIF / IMAGE / VIDEO)
    # ------------------------------------------
    if active_mode == "gif":
        if len(current_gif_frames) > 0:
            frame_to_show = current_gif_frames[current_gif_index]

            current_gif_index += 1

            if current_gif_index >= len(current_gif_frames):
                # GIF finished → go idle
                set_idle_state()

            frame_to_show = cv2.resize(frame_to_show, (cam_w, cam_h))
        else:
            frame_to_show = np.zeros((cam_h, cam_w, 3), dtype=np.uint8)


    elif active_mode == "image":
        if len(current_gif_frames) > 0:
            frame_to_show = cv2.resize(current_gif_frames[0], (cam_w, cam_h))
        else:
            frame_to_show = np.zeros((cam_h, cam_w, 3), dtype=np.uint8)


    elif active_mode == "video":
        if not video_queue.empty():
            frame_to_show = cv2.resize(video_queue.get(), (cam_w, cam_h))
        elif video_done_flag.is_set():
            set_idle_state()
            frame_to_show = cv2.resize(IMG_IDLE_STARE, (cam_w, cam_h))
        else:
            frame_to_show = np.zeros((cam_h, cam_w, 3), dtype=np.uint8)

    elif active_mode == "idle":
        frame_to_show = cv2.resize(IMG_IDLE_STARE, (cam_w, cam_h))

    else:
        frame_to_show = np.zeros((cam_h, cam_w, 3), dtype=np.uint8)


    # ------------------------------------------
    # Combine webcam (left) + meme (right)
    # ------------------------------------------
    # --- SAFETY NORMALIZATION FOR frame_to_show ---

    # If frame_to_show is None or not an array, make it a black image
    if frame_to_show is None or not isinstance(frame_to_show, np.ndarray):
        frame_to_show = np.zeros_like(resized_cam)

    # If it's 2D (grayscale), convert to BGR
    if len(frame_to_show.shape) == 2:
        frame_to_show = cv2.cvtColor(frame_to_show, cv2.COLOR_GRAY2BGR)

    # If it's 4-channel (e.g. from some PNG/GIF with alpha), drop alpha
    if frame_to_show.shape[2] == 4:
        frame_to_show = frame_to_show[:, :, :3]

    # Finally, resize to match webcam size
    frame_to_show = cv2.resize(frame_to_show, (resized_cam.shape[1], resized_cam.shape[0]))

    combined = cv2.hconcat([resized_cam, frame_to_show])
    cv2.imshow("Freak Detector", combined)


    # Quit on ESC
    if cv2.waitKey(1) & 0xFF == 27:
        break


cap.release()
cv2.destroyAllWindows()
