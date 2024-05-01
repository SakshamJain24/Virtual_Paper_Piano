import pygame

# Dictionary to store loaded sound files
sound_files = {}

def load_sound_files(tiles):
    pygame.init()
    for tile_name in tiles:
        sound_file = f"Sounds/{tile_name}.mp3"
        sound_files[tile_name] = pygame.mixer.Sound(sound_file)

def play_sound(tile_name):
    global sound_files
    # Stop any currently playing sound
    pygame.mixer.stop()
    # Play the sound if it exists in the dictionary
    if tile_name in sound_files:
        sound_files[tile_name].play()
