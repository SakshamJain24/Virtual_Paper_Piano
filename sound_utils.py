import os
from playsound import playsound

# Global variable to store sound files
sound_files = {}


def load_sound_files(tiles):
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
