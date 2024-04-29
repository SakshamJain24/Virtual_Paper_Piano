import cv2
import numpy as np


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

    return labeled_image, tiles  # Return both the labeled image and the tiles dictionary
