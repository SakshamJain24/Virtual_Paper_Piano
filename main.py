import cv2
import threading
from detect_and_label_corners import detect_and_label_corners
from hand_tracking import hand_tracking
from sound_utils import load_sound_files
#Applied changes to the main file as well

# Global variables
fixed_corners = []
tiles = {}


def main():
    cap = cv2.VideoCapture(0)

    # Stage 1: Capture corner points and assign tiles
    capturing_corners = True
    while capturing_corners:
        ret, frame = cap.read()

        if not ret:
            break

        labeled_image, tiles = detect_and_label_corners(frame.copy())

        cv2.imshow('Image with Labeled Corners and Tiles', labeled_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            capturing_corners = False

    load_sound_files(tiles)

    # Start hand tracking in a separate thread
    hand_thread = threading.Thread(target=hand_tracking, args=(cap, tiles))
    hand_thread.start()

    # Continue processing frames in the main thread
    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # Perform any additional processing or display as needed

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Wait for the hand tracking thread to finish
    hand_thread.join()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()