import cv2
import numpy as np
import mediapipe as mp
from sound_utils import play_sound


def hand_tracking(cap, tiles):
    # Initialize MediaPipe Hands model
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()

    while True:
        # Read frame from the webcam
        ret, frame = cap.read()

        if not ret:
            break

        # Convert the image to RGB before feeding it to MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the image with MediaPipe Hands model
        results = hands.process(frame_rgb)

        # Check if hands are detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Extract landmarks for each hand
                for landmark in hand_landmarks.landmark:
                    # Convert normalized coordinates to pixel coordinates
                    image_height, image_width, _ = frame.shape
                    x, y = int(landmark.x * image_width), int(landmark.y * image_height)

                    # Check if fingertip is within any tile region
                    for tile_name, tile_corners in tiles.items():
                        # Assuming tile_corners are in clockwise order
                        if cv2.pointPolygonTest(np.array(tile_corners), (x, y), False) >= 0:
                            print(f"Finger tip is over {tile_name}")
                            # Play sound associated with the tile
                            play_sound(tile_name)

        # Display the image with detected hands and fingers
        cv2.imshow('Hand Tracking', frame)

        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam
    cap.release()
