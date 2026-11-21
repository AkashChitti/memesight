"""Checks for MemeSight's gesture logic, debounce and playback. No webcam needed.

Run with: python test_gestures.py
"""
import os
import tempfile
import time
from types import SimpleNamespace

import numpy as np

import memesight as m


def check(name, condition):
    assert condition, f"FAILED: {name}"
    print(f"ok  {name}")


def L(x=0.5, y=0.5):
    return SimpleNamespace(x=x, y=y)


def hand():
    return SimpleNamespace(landmark=[L() for _ in range(21)])


def face(n=478):
    return [L() for _ in range(n)]


# ---------------------------------------------------------------- media
check("no missing media", m.MISSING_MEDIA == [])
for clip in (m.GIF_TONGUE_SHAKE, m.GIF_HEAD_NOD, m.GIF_HANDS_ON_HEAD):
    check("gif clip has frames", len(clip) > 0)
    check("gif frame fits panel", all(
        frame.shape[0] <= m.PANEL_H and frame.shape[1] <= m.PANEL_W for frame, _ in clip
    ))
    check("gif has positive delays", all(delay > 0 for _, delay in clip))
check("stills loaded", all(img is not None for img in
      (m.IMG_HAND_ON_CHIN, m.IMG_ONE_FINGER_UP, m.IMG_IDLE_STARE, m.IMG_THUMBS_UP)))

# ------------------------------------------------------- head gestures
check("reversals counted", m.count_reversals([0, 10, 0, 10, 0]) == 3)
check("monotonic has none", m.count_reversals([0, 1, 2, 3]) == 0)
check("jitter ignored", m.count_reversals([1.0, 1.0001, 0.9999, 1.0]) == 0)

m.nose_x_buffer.clear()
m.nose_x_buffer.extend(np.linspace(0.40, 0.60, 15))                  # one sweep
check("single sweep is not a shake", m.detect_head_shake() is False)
m.nose_x_buffer.clear()
m.nose_x_buffer.extend([0.40, 0.50, 0.60, 0.50, 0.40] * 3)          # back and forth
check("oscillation is a shake", m.detect_head_shake() is True)

m.nose_y_buffer.clear()
m.nose_y_buffer.extend(np.linspace(0.40, 0.60, 15))
check("single drift is not a nod", m.detect_head_nod() is False)
m.nose_y_buffer.clear()
m.nose_y_buffer.extend([0.40, 0.50, 0.60, 0.50, 0.40] * 4)
check("oscillation is a nod", m.detect_head_nod() is True)

# ---------------------------------------------------------- mouth open
f = face()
f[13], f[14], f[16] = L(y=0.40), L(y=0.45), L(y=0.47)
check("mouth open", m.detect_tongue(f) is True)
f[16] = L(y=0.44)
check("closed mouth rejected", m.detect_tongue(f) is False)

# ------------------------------------------------------ hand gestures
f = face()
f[199] = L(0.5, 0.5)
h = hand()
h.landmark[8] = L(0.52, 0.52)
check("hand on chin", m.detect_hand_on_chin(f, h) is True)
h.landmark[8] = L(0.9, 0.9)
check("far hand is not chin", m.detect_hand_on_chin(f, h) is False)

h = hand()
h.landmark[8], h.landmark[7] = L(0.5, 0.2), L(0.5, 0.4)
h.landmark[12], h.landmark[11] = L(0.6, 0.6), L(0.6, 0.4)
h.landmark[16], h.landmark[15] = L(0.4, 0.6), L(0.4, 0.4)
h.landmark[20], h.landmark[19] = L(0.3, 0.6), L(0.3, 0.4)
check("one finger up", m.detect_one_finger_up(h) is True)
h.landmark[12].y = 0.2
check("open hand is not one finger", m.detect_one_finger_up(h) is False)

h = hand()
h.landmark[4], h.landmark[3], h.landmark[2] = L(y=0.10), L(y=0.20), L(y=0.30)
for tip, pip in ((8, 6), (12, 10), (16, 14), (20, 18)):
    h.landmark[tip], h.landmark[pip] = L(y=0.60), L(y=0.50)
check("thumbs up", m.detect_thumbs_up(h) is True)
h.landmark[8].y = 0.20
check("index out is not thumbs up", m.detect_thumbs_up(h) is False)

f = face()
f[10] = L(0.5, 0.4)
a, b = hand(), hand()
a.landmark[8] = L(0.35, 0.30)
b.landmark[8] = L(0.65, 0.30)
check("hands on head", m.detect_hands_on_head(f, [a, b]) is True)
check("one hand is not both", m.detect_hands_on_head(f, [a]) is False)
a.landmark[8] = L(0.05, 0.30)
check("hands beside head rejected", m.detect_hands_on_head(f, [a, b]) is False)
a.landmark[8] = L(0.35, 0.60)
check("hands below forehead rejected", m.detect_hands_on_head(f, [a, b]) is False)

a, b = hand(), hand()
a.landmark[9] = L(0.40, 0.5)
rubbing = False
for i in range(22):
    b.landmark[9] = L(0.45 if i % 2 == 0 else 0.60, 0.5)
    rubbing = m.detect_rubbing_palms([a, b]) or rubbing
check("rubbing palms", rubbing is True)
check("one hand clears buffer",
      m.detect_rubbing_palms([a]) is False and len(m.palm_dist_buffer) == 0)

# -------------------------------------------------------- idle + gaze
m.nose_x_buffer.clear()
m.nose_y_buffer.clear()
m.nose_x_buffer.extend([0.5] * 15)
m.nose_y_buffer.extend([0.5] * 15)
f = face()
f[33], f[133], f[468] = L(0.3), L(0.4), L(0.35)     # right iris centred
f[362], f[263], f[473] = L(0.6), L(0.7), L(0.65)    # left iris centred
check("idle stare", m.detect_idle_stare(f) is True)
f[468] = L(0.395)                                    # right iris far to the side
check("looking aside is not idle", m.detect_idle_stare(f) is False)
f[468] = L(0.35)
m.nose_x_buffer.extend([0.53])                       # head moving instead
check("moving head is not idle", m.detect_idle_stare(f) is False)

# ------------------------------------------------------------ priority
check("priority picks tongue", m.evaluate_gesture_priority(1, 1, 1, 1, 1, 1, 1, 1) == "tongue_shake")
check("priority picks thumbs", m.evaluate_gesture_priority(0, 0, 0, 0, 1, 1, 1, 1) == "thumbs_up")
check("priority empty", m.evaluate_gesture_priority(0, 0, 0, 0, 0, 0, 0, 0) == "none")

# ------------------------------------------------------------- debounce
gate = m.TriggerGate(hold_seconds=0.2, cooldown_seconds=1.0)
check("needs hold", gate.update("head_nod", 0.0) is None)
check("not yet held", gate.update("head_nod", 0.1) is None)
check("fires after hold", gate.update("head_nod", 0.25) == "head_nod")
check("silent during cooldown", gate.update("head_nod", 0.5) is None)
check("still silent near cooldown end", gate.update("head_nod", 1.2) is None)
check("re-triggers after cooldown", gate.update("head_nod", 1.3) is None)
check("re-fires when held again", gate.update("head_nod", 1.55) == "head_nod")

gate = m.TriggerGate(0.2, 1.0)
gate.update("head_nod", 0.0)
gate.update("head_nod", 0.25)
check("cooldown is not reset by a new key", gate.update("idle", 0.9) is None)

# -------------------------------------------------------------- to_panel
check("None becomes black panel",
      m.to_panel(None).shape == (m.PANEL_H, m.PANEL_W, 3))
gray = m.to_panel(np.zeros((10, 10), np.uint8))
check("gray widened to BGR", gray.shape == (m.PANEL_H, m.PANEL_W, 3))
rgba = m.to_panel(np.zeros((10, 10, 4), np.uint8))
check("alpha dropped", rgba.shape == (m.PANEL_H, m.PANEL_W, 3))
check("small frames resized", m.to_panel(np.zeros((10, 10, 3), np.uint8)).shape[:2]
      == (m.PANEL_H, m.PANEL_W))

# ------------------------------------------------------------ playback
panel = m.MemePanel()
check("starts idle", panel.frame(0.0) is m.IMG_IDLE_STARE)

first = np.zeros((10, 10, 3), np.uint8)
second = np.ones((10, 10, 3), np.uint8)
panel.show_gif([(first, 0.1), (second, 0.1)])
panel.next_frame_at = 10.0
check("gif shows first frame", panel.frame(9.0) is first)
check("gif holds until due", panel.frame(9.99) is first)
check("gif advances on time", panel.frame(10.0) is second)
check("gif holds second frame", panel.frame(10.05) is second)
check("gif ends in idle", panel.frame(10.2) is m.IMG_IDLE_STARE and panel.mode == "idle")

tmp = os.path.join(tempfile.mkdtemp(), "clip.mp4")
writer = m.cv2.VideoWriter(tmp, m.cv2.VideoWriter_fourcc(*"mp4v"), 30, (320, 240))
if writer.isOpened():
    for i in range(30):
        writer.write(np.full((240, 320, 3), i * 8, np.uint8))
    writer.release()

    panel.play_video(tmp)
    started = time.time()
    seen = set()
    while time.time() - started < 3 and panel.mode == "video":
        frame = panel.frame()
        if frame is not None:
            seen.add(round(float(frame.mean()) / 8))
        time.sleep(0.005)
    elapsed = time.time() - started
    check("video played to the end", len(seen) >= 27 and panel.mode == "idle")
    check("video paced near real time", 0.6 <= elapsed <= 2.0)
    print(f"    {len(seen)} distinct frames in {elapsed:.2f}s")

    # Switching clips must stop the previous reader immediately.
    other = os.path.join(tempfile.mkdtemp(), "other.mp4")
    writer = m.cv2.VideoWriter(other, m.cv2.VideoWriter_fourcc(*"mp4v"), 30, (320, 240))
    for _ in range(60):
        writer.write(np.full((240, 320, 3), 200, np.uint8))
    writer.release()

    panel.play_video(tmp)
    time.sleep(0.2)
    panel.play_video(other)
    stale = 0
    started = time.time()
    while time.time() - started < 0.6:
        frame = panel.frame()
        if frame is not None and int(frame.mean()) < 100:
            stale += 1
        time.sleep(0.005)
    check("no frames leak from the replaced clip", stale == 0)
    panel.stop_video()
else:
    print("skip  video playback (this OpenCV build has no mp4 writer)")

# --------------------------------------------------- end-to-end loop
class FakeCamera:
    def isOpened(self):
        return True

    def read(self):
        return True, np.full((480, 640, 3), 40, np.uint8)

    def release(self):
        pass


shown = []
calls = {"n": 0}


def fake_wait_key(_delay):
    calls["n"] += 1
    return 27 if calls["n"] >= 4 else -1


m.cv2.VideoCapture = lambda *a, **k: FakeCamera()
m.cv2.imshow = lambda _name, img: shown.append(img.shape)
m.cv2.waitKey = fake_wait_key
m.cv2.getWindowProperty = lambda *a, **k: 1
m.cv2.destroyAllWindows = lambda *a, **k: None

m.main()
check("loop drew split screen", shown == [(m.PANEL_H, m.PANEL_W * 2, 3)] * 4)
check("loop exited", calls["n"] == 4)

print("\nALL CHECKS PASSED")
