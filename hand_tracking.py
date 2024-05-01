import cv2
import numpy as np
import mediapipe as mp
from sound_utils import play_sound

# Variable to store the currently active tile
active_tile = None

def hand_tracking(cap, tiles):
    global active_tile
    mp_hands = mp.solutions.hands.Hands()

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = mp_hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for landmark in hand_landmarks.landmark:
                    image_height, image_width, _ = frame.shape
                    x, y = int(landmark.x * image_width), int(landmark.y * image_height)
                    for tile_name, tile_corners in tiles.items():
                        if cv2.pointPolygonTest(np.array(tile_corners), (x, y), False) >= 0:
                            if active_tile != tile_name:
                                print(f"Finger tip is over {tile_name}")
                                play_sound(tile_name)
                                active_tile = tile_name

        cv2.imshow('Hand Tracking', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
