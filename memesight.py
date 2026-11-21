"""MemeSight AI — webcam gestures trigger memes in a split-screen window."""

import queue
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image

# ==========================================
#                 SETTINGS
# ==========================================

PANEL_W, PANEL_H = 640, 480
WINDOW_NAME = "MemeSight AI"
MEME_DIR = Path(__file__).resolve().parent / "memes"

COOLDOWN_SECONDS = 1.0        # ignore new gestures right after a trigger
GESTURE_HOLD_SECONDS = 0.2    # how long a gesture must persist
IDLE_STARE_SECONDS = 1.5      # stillness before the idle meme
GIF_MIN_FRAME_SECONDS = 0.02  # floor for GIFs with broken frame durations
CAMERA_INDEX = 0

PATH_TONGUE_SHAKE      = MEME_DIR / "freaky-orca.gif"
PATH_HAND_ON_CHIN      = MEME_DIR / "monkeythink.jpg"
PATH_HEAD_NOD          = MEME_DIR / "ishowspeed.gif"
PATH_ONE_FINGER_UP     = MEME_DIR / "monkeyrealize.jpeg"
PATH_HANDS_ON_HEAD     = MEME_DIR / "ishowspeed-wow.gif"
PATH_RUBBING_PALMS     = MEME_DIR / "freaky-sonic.mp4"
PATH_THUMBS_UP         = MEME_DIR / "thumbsupmonkey.png"
PATH_IDLE_STARE        = MEME_DIR / "monkeytruth.jpg"

# ==========================================
#            MEDIA LOADING
# ==========================================


def load_gif_clip(path):
    """Return [(frame, seconds)] for a GIF, downscaling frames larger than the panel.

    A large GIF expands to hundreds of megabytes once decoded, and the panel only
    ever shows PANEL_W x PANEL_H.
    """
    clip = []
    try:
        gif = Image.open(path)
    except OSError:
        return clip

    while True:
        try:
            gif.seek(len(clip))
        except EOFError:
            break

        bgr = cv2.cvtColor(np.array(gif.convert("RGB")), cv2.COLOR_RGB2BGR)
        if bgr.shape[1] > PANEL_W or bgr.shape[0] > PANEL_H:
            bgr = cv2.resize(bgr, (PANEL_W, PANEL_H))

        delay = gif.info.get("duration", 0) / 1000.0
        clip.append((bgr, max(delay, GIF_MIN_FRAME_SECONDS)))

    return clip


GIF_TONGUE_SHAKE  = load_gif_clip(PATH_TONGUE_SHAKE)
GIF_HEAD_NOD      = load_gif_clip(PATH_HEAD_NOD)
GIF_HANDS_ON_HEAD = load_gif_clip(PATH_HANDS_ON_HEAD)

IMG_HAND_ON_CHIN  = cv2.imread(str(PATH_HAND_ON_CHIN))
IMG_ONE_FINGER_UP = cv2.imread(str(PATH_ONE_FINGER_UP))
IMG_IDLE_STARE    = cv2.imread(str(PATH_IDLE_STARE))
IMG_THUMBS_UP     = cv2.imread(str(PATH_THUMBS_UP))

MISSING_MEDIA = [
    path.name
    for path, asset in (
        (PATH_TONGUE_SHAKE, GIF_TONGUE_SHAKE),
        (PATH_HEAD_NOD, GIF_HEAD_NOD),
        (PATH_HANDS_ON_HEAD, GIF_HANDS_ON_HEAD),
        (PATH_HAND_ON_CHIN, IMG_HAND_ON_CHIN),
        (PATH_ONE_FINGER_UP, IMG_ONE_FINGER_UP),
        (PATH_IDLE_STARE, IMG_IDLE_STARE),
        (PATH_THUMBS_UP, IMG_THUMBS_UP),
    )
    if asset is None or (isinstance(asset, list) and not asset)
]

if MISSING_MEDIA:
    raise SystemExit(
        f"❌ Missing or unreadable meme files in {MEME_DIR}: {', '.join(MISSING_MEDIA)}"
    )

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

# Buffers of recent samples. They only hold frames where the relevant model fired,
# so every detector clears its buffer when its input disappears.
nose_x_buffer = deque(maxlen=15)
nose_y_buffer = deque(maxlen=20)
palm_dist_buffer = deque(maxlen=20)


# ==========================================
#       GESTURE DETECTION FUNCTIONS
# ==========================================

def count_reversals(values):
    """How many times a sample sequence changes direction (ignoring tiny jitter)."""
    samples = list(values)
    reversals = 0
    last_sign = 0
    for previous, current in zip(samples, samples[1:]):
        delta = current - previous
        if abs(delta) < 0.001:
            continue
        sign = 1 if delta > 0 else -1
        if last_sign and sign != last_sign:
            reversals += 1
        last_sign = sign
    return reversals


# 1. Mouth open (FaceMesh has no tongue landmark, so "tongue out" is really
#    an open mouth) + head shake.
def detect_tongue(lm):
    upper = lm[13].y
    lower = lm[14].y
    tongue = lm[16].y
    return (lower - upper) > 0.01 and (tongue - lower) > 0.01


# 2. Head SHAKE: sideways motion that actually reverses direction.
def detect_head_shake():
    if len(nose_x_buffer) < 15:
        return False
    motion = max(nose_x_buffer) - min(nose_x_buffer)
    return motion > 0.03 and count_reversals(nose_x_buffer) >= 2


# 3. Head NOD: vertical motion that actually reverses direction.
def detect_head_nod():
    if len(nose_y_buffer) < 15:
        return False
    motion = max(nose_y_buffer) - min(nose_y_buffer)
    return motion > 0.07 and count_reversals(nose_y_buffer) >= 2


# 4. Hand ON CHIN.
def detect_hand_on_chin(face_lm, hand_lm):
    chin = face_lm[199]
    tip = hand_lm.landmark[8]  # index finger tip

    dx = chin.x - tip.x
    dy = chin.y - tip.y

    return (dx * dx + dy * dy) ** 0.5 < 0.12


# 5. One finger pointing UP.
def detect_one_finger_up(hand_lm):
    index_up = hand_lm.landmark[8].y < hand_lm.landmark[7].y
    mid_down = hand_lm.landmark[12].y > hand_lm.landmark[11].y
    ring_down = hand_lm.landmark[16].y > hand_lm.landmark[15].y
    pinky_down = hand_lm.landmark[20].y > hand_lm.landmark[19].y

    return index_up and mid_down and ring_down and pinky_down


# 6. Both hands ON HEAD: above the forehead and within head width of its centre.
def detect_hands_on_head(face_lm, hand_list):
    if len(hand_list) < 2:
        return False

    forehead = face_lm[10]

    for hand in hand_list:
        tip = hand.landmark[8]
        if tip.y >= forehead.y + 0.02:
            return False
        if abs(tip.x - forehead.x) > 0.30:
            return False
    return True


# 7. Palms rubbing: close together AND oscillating back and forth.
def detect_rubbing_palms(hand_list):
    # Clearing on separation stops stale distances from an earlier two-hand frame
    # from faking oscillation.
    if len(hand_list) < 2:
        palm_dist_buffer.clear()
        return False

    p1 = hand_list[0].landmark[9]   # palm centre
    p2 = hand_list[1].landmark[9]

    dx = p1.x - p2.x
    dy = p1.y - p2.y
    dist = (dx * dx + dy * dy) ** 0.5
    palm_dist_buffer.append(dist)

    if len(palm_dist_buffer) < 20:
        return False

    oscillation = max(palm_dist_buffer) - min(palm_dist_buffer)
    return oscillation > 0.06 and count_reversals(palm_dist_buffer) >= 3 and dist < 0.18


# 8. Thumbs up: thumb points up, every other finger is curled.
def detect_thumbs_up(hand_lm):
    lm = hand_lm.landmark

    thumb_up = lm[4].y < lm[3].y < lm[2].y
    fingers_curled = all(lm[tip].y > lm[pip].y for tip, pip in
                         ((8, 6), (12, 10), (16, 14), (20, 18)))
    return thumb_up and fingers_curled


# 9. Idle stare: head still and irises centred in their eyes.
def gaze_ratio(iris, inner_corner, outer_corner):
    span = outer_corner.x - inner_corner.x
    if abs(span) < 1e-6:
        return 0.5
    return (iris.x - inner_corner.x) / span


def detect_idle_stare(face_lm):
    if len(nose_x_buffer) < 15 or len(nose_y_buffer) < 15:
        return False

    stable_x = (max(nose_x_buffer) - min(nose_x_buffer)) < 0.005
    stable_y = (max(nose_y_buffer) - min(nose_y_buffer)) < 0.005
    if not (stable_x and stable_y):
        return False

    # Iris landmarks only exist with refine_landmarks; without them, a still head
    # is all we can measure.
    if len(face_lm) <= 473:
        return True

    left = gaze_ratio(face_lm[468], face_lm[133], face_lm[33])
    right = gaze_ratio(face_lm[473], face_lm[362], face_lm[263])
    return 0.2 < left < 0.8 and 0.2 < right < 0.8


# ==========================================
#         PRIORITY ORDER FOR GESTURES
# ==========================================

def evaluate_gesture_priority(
        tongue_shake,
        hand_chin,
        head_nod,
        one_finger_up,
        thumbs_up,
        hands_on_head,
        rubbing_palms_flag,
        idle_stare
    ):
    """Return the key of the highest-priority gesture that fired, or "none"."""

    if tongue_shake:
        return "tongue_shake"
    if hand_chin:
        return "hand_chin"
    if head_nod:
        return "head_nod"
    if one_finger_up:
        return "one_finger_up"
    if thumbs_up:
        return "thumbs_up"
    if hands_on_head:
        return "hands_on_head"
    if rubbing_palms_flag:
        return "rubbing_palms"
    if idle_stare:
        return "idle"
    return "none"


# ==========================================
#        TRIGGER DEBOUNCE
# ==========================================

class TriggerGate:
    """Turns a per-frame gesture key into a debounced trigger.

    A gesture must stay top-priority for hold_seconds, then no trigger is accepted
    for cooldown_seconds. Both are wall-clock, so behaviour does not change with FPS.
    """

    def __init__(self, hold_seconds, cooldown_seconds):
        self.hold_seconds = hold_seconds
        self.cooldown_seconds = cooldown_seconds
        self._since = None
        self._key = None
        self._blocked_until = 0.0

    def update(self, key, now):
        """Return the gesture to fire, or None."""
        if now < self._blocked_until or key == "none":
            self._since = None
            self._key = None
            return None

        if key != self._key:
            self._key = key
            self._since = now
            return None

        if now - self._since >= self.hold_seconds:
            self._blocked_until = now + self.cooldown_seconds
            self._since = None
            self._key = None
            return key

        return None


# ==========================================
#        PLAYBACK PANEL (RIGHT HALF)
# ==========================================

class MemePanel:
    """Owns what the right half shows and when an animated clip advances."""

    def __init__(self):
        self.mode = "idle"          # idle / image / gif / video
        self.clip = [(IMG_IDLE_STARE, 0.0)]
        self.index = 0
        self.next_frame_at = 0.0
        self.frames = queue.Queue(maxsize=2)
        self.done = threading.Event()
        self.token = 0
        self.last_video_frame = None
        self.video_path = None
        self.lock = threading.Lock()

    def show_gif(self, clip):
        # Re-triggering the gesture that is already playing must not restart it.
        if self.mode == "gif" and self.clip is clip:
            return
        self.stop_video()
        self.mode = "gif"
        self.clip = clip or [(IMG_IDLE_STARE, 0.0)]
        self.index = 0
        self.next_frame_at = time.monotonic() + self.clip[0][1]

    def show_image(self, img):
        self.stop_video()
        self.mode = "image"
        self.clip = [(img, 0.0)]
        self.index = 0

    def show_idle(self):
        self.stop_video()
        self.mode = "idle"
        self.clip = [(IMG_IDLE_STARE, 0.0)]
        self.index = 0

    def play_video(self, path):
        # Re-triggering the gesture that is already playing must not restart it.
        if self.mode == "video" and self.video_path == path and not self.done.is_set():
            return
        self.stop_video()
        with self.lock:
            # Token changes and queue access share the lock, so a reader blocked
            # on a full queue can never enqueue a frame after this switch.
            self.mode = "video"
            self.video_path = path
            self.last_video_frame = None
            self.done.clear()
            self.token += 1
            token = self.token
        threading.Thread(
            target=self._read_video, args=(path, token), daemon=True
        ).start()

    def stop_video(self):
        with self.lock:
            self.token += 1
            self._drop_frames()
            self.done.set()
            self.video_path = None

    def _drop_frames(self):
        while True:
            try:
                self.frames.get_nowait()
            except queue.Empty:
                return

    def _current(self, token):
        return token == self.token and self.mode == "video"

    def _queue_frame(self, frame, token):
        """Block until this frame is queued, giving up if the clip changed."""
        while True:
            with self.lock:
                if not self._current(token):
                    return False
                try:
                    self.frames.put_nowait(frame)
                    return True
                except queue.Full:
                    pass
            time.sleep(0.005)

    def _read_video(self, path, token):
        cap = cv2.VideoCapture(str(path))
        try:
            if not cap.isOpened():
                print(f"❌ Unable to open video: {path}")
                return

            fps = cap.get(cv2.CAP_PROP_FPS)
            delay = 1.0 / fps if 1 <= fps <= 120 else 1.0 / 30

            while self._current(token):
                started = time.monotonic()
                ok, frame = cap.read()
                if not ok:
                    break

                if not self._queue_frame(frame, token):
                    break

                time.sleep(max(0.0, delay - (time.monotonic() - started)))
        finally:
            cap.release()
            if token == self.token:
                self.done.set()

    def frame(self, now=None):
        """Return the panel image for display now (None means "nothing yet")."""
        now = time.monotonic() if now is None else now

        if self.mode == "video":
            try:
                self.last_video_frame = self.frames.get_nowait()
            except queue.Empty:
                if self.done.is_set():
                    self.show_idle()
                    return IMG_IDLE_STARE
            return self.last_video_frame

        if self.mode == "gif":
            if now >= self.next_frame_at:
                if self.index + 1 >= len(self.clip):
                    self.show_idle()
                    return IMG_IDLE_STARE
                self.index += 1
                self.next_frame_at = now + self.clip[self.index][1]
            return self.clip[self.index][0]

        return self.clip[0][0]


def to_panel(img):
    """Normalize any meme asset to a PANEL_W x PANEL_H BGR frame."""
    if img is None:
        return np.zeros((PANEL_H, PANEL_W, 3), dtype=np.uint8)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        img = img[:, :, :3]
    if img.shape[1] != PANEL_W or img.shape[0] != PANEL_H:
        img = cv2.resize(img, (PANEL_W, PANEL_H))
    return img


# ==========================================
#                 MAIN LOOP
# ==========================================

def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise SystemExit(f"❌ Could not open webcam (device {CAMERA_INDEX}).")

    panel = MemePanel()
    gate = TriggerGate(GESTURE_HOLD_SECONDS, COOLDOWN_SECONDS)
    idle_since = None

    actions = {
        "tongue_shake": lambda: panel.show_gif(GIF_TONGUE_SHAKE),
        "hand_chin": lambda: panel.show_image(IMG_HAND_ON_CHIN),
        "head_nod": lambda: panel.show_gif(GIF_HEAD_NOD),
        "one_finger_up": lambda: panel.show_image(IMG_ONE_FINGER_UP),
        "thumbs_up": lambda: panel.show_image(IMG_THUMBS_UP),
        "hands_on_head": lambda: panel.show_gif(GIF_HANDS_ON_HEAD),
        "rubbing_palms": lambda: panel.play_video(PATH_RUBBING_PALMS),
        "idle": panel.show_idle,
    }

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("❌ Lost the webcam feed.")
                break

            now = time.monotonic()
            frame = cv2.flip(frame, 1)
            resized_cam = cv2.resize(frame, (PANEL_W, PANEL_H))

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
                nose_x_buffer.append(face_lm[1].x)
                nose_y_buffer.append(face_lm[1].y)

            if hands_result.multi_hand_landmarks:
                hand_lms = hands_result.multi_hand_landmarks

            # ------------------------------------------
            # Detect all gestures
            # ------------------------------------------
            if face_lm:
                tongue_shake = detect_tongue(face_lm) and detect_head_shake()
                nod = detect_head_nod()
                still = detect_idle_stare(face_lm)
            else:
                tongue_shake = nod = still = False
                # Drop stale samples so re-entry cannot fake a nod or shake.
                nose_x_buffer.clear()
                nose_y_buffer.clear()

            # Idle physically requires a still, hand-free face.
            if still and not hand_lms:
                idle_since = idle_since or now
            else:
                idle_since = None
            idle = idle_since is not None and now - idle_since >= IDLE_STARE_SECONDS

            hand_chin = False
            one_finger_up = False
            thumbs_up = False
            hands_on_head = False

            if hand_lms:
                if len(hand_lms) == 1:
                    hl = hand_lms[0]
                    if face_lm:
                        hand_chin = detect_hand_on_chin(face_lm, hl)
                    one_finger_up = detect_one_finger_up(hl)
                    thumbs_up = detect_thumbs_up(hl)

                # Keep the palm buffer fresh even while the face is out of frame.
                rubbing_palms_flag = detect_rubbing_palms(hand_lms)

                if len(hand_lms) >= 2 and face_lm:
                    hands_on_head = detect_hands_on_head(face_lm, hand_lms)
            else:
                rubbing_palms_flag = detect_rubbing_palms(hand_lms)

            # ------------------------------------------
            # Priority engine + debounce
            # ------------------------------------------
            highest = evaluate_gesture_priority(
                tongue_shake,
                hand_chin,
                nod,
                one_finger_up,
                thumbs_up,
                hands_on_head,
                rubbing_palms_flag,
                idle
            )

            fired = gate.update(highest, now)
            if fired:
                print("🎯 Trigger:", fired)
                actions[fired]()

            # ------------------------------------------
            # Combine webcam (left) + meme (right)
            # ------------------------------------------
            combined = cv2.hconcat([resized_cam, to_panel(panel.frame(now))])
            cv2.imshow(WINDOW_NAME, combined)

            # Quit on ESC or when the window is closed
            if cv2.waitKey(1) & 0xFF == 27:
                break
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        cap.release()
        panel.stop_video()
        face_mesh.close()
        hands.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
