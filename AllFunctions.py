import cv2
import numpy as np
import os
import threading
from playsound import playsound  # Import playsound for playing audio files
import mediapipe as mp
# from detect_and_label_corners import detect_and_label_corners

# Global variables
fixed_corners = []
tiles = {}
sound_files = {}

def detect_and_label_corners(image):
    global fixed_corners
    global tiles

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect corners using Shi-Tomasi corner detection
    corners = cv2.goodFeaturesToTrack(gray, maxCorners=20, qualityLevel=0.01, minDistance=15)

    # If no corners are found, return the original image
    if corners is None:
        return image

    # Convert corners to integers
    corners = np.int0(corners)

    # Sort corners by x-coordinate
    corners = sorted(corners, key=lambda x: x[0][0])

    # Store fixed corner positions
    fixed_corners = [corner.ravel() for corner in corners]

    # Initialize the tiles dictionary
    tiles = {}

    # Iterate over the fixed corners to create tiles
    num_tiles = len(fixed_corners) // 2 - 1
    for i in range(num_tiles):
        # Calculate the indices for the corners of the tile
        start_index = i * 2
        end_index = start_index + 4

        # Extract the corners for the current tile
        tile_corners = [fixed_corners[start_index], fixed_corners[start_index + 1],
                        fixed_corners[start_index + 3], fixed_corners[start_index + 2]]

        # Assign the corners to the tile in the desired order
        tiles[f"Tile {i + 1}"] = tile_corners

    # Draw circles at the corner positions and label them sequentially
    labeled_image = image.copy()
    for i, corner in enumerate(fixed_corners):
        x, y = corner
        cv2.circle(labeled_image, (x, y), 5, (0, 255, 0), -1)
        cv2.putText(labeled_image, f"Corner {i + 1}", (x + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Draw tiles on the image
    for tile, corners in tiles.items():
        # Convert corners to a numpy array
        corners = np.array(corners).reshape((-1, 1, 2))
        # Draw the polygon representing the tile
        cv2.polylines(labeled_image, [corners], isClosed=True, color=(0, 0, 255), thickness=2)
        # Display the name of the tile
        text_position = (int(np.mean(corners[:, :, 0])), int(np.mean(corners[:, :, 1])))
        cv2.putText(labeled_image, tile, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    return labeled_image


def load_sound_files():
    global sound_files
    sounds_folder = "Sounds"  # Folder containing sound files
    for tile_name in tiles:
        sound_files[tile_name] = os.path.join(sounds_folder, f"{tile_name}.mp3")


def play_sound(tile_name):
    global sound_files
    # Load the sound file if not already loaded
    if tile_name not in sound_files:
        return
    playsound(sound_files[tile_name])


def hand_tracking(cap):
    global sound_files

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

                    # Check if finger tip is within any tile region
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


def main():
    global fixed_corners
    global tiles

    # Start video capture from webcam
    cap = cv2.VideoCapture(0)

    # Stage 1: Capture corner points and assign tiles
    capturing_corners = True
    while capturing_corners:
        # Read frame from the webcam
        ret, frame = cap.read()

        if not ret:
            break

        # Detect and label corners, and assign tiles
        labeled_image = detect_and_label_corners(frame.copy())  # Ensure we're working with a copy of the image

        # Display the image with detected corners and tiles
        cv2.imshow('Image with Labeled Corners and Tiles', labeled_image)

        # Break the loop and proceed to hand tracking if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            capturing_corners = False

    # Load sound files into memory
    load_sound_files()

    # Start hand tracking in a separate thread
    hand_thread = threading.Thread(target=hand_tracking, args=(cap,))
    hand_thread.start()

    # Wait for the hand tracking thread to finish
    hand_thread.join()

    # Close OpenCV windows
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
