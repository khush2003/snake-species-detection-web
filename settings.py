from pathlib import Path
import sys

# Get the absolute path of the current file
file_path = Path(__file__).resolve()

# Get the parent directory of the current file
root_path = file_path.parent

# Add the root path to the sys.path list if it is not already there
if root_path not in sys.path:
    sys.path.append(str(root_path))

# Get the relative path of the root directory with respect to the current working directory
ROOT = root_path.relative_to(Path.cwd())

# Sources
IMAGE = 'Image'
VIDEO = 'Video'
WEBCAM = 'Webcam'
YOUTUBE = 'YouTube'

SOURCES_LIST = [IMAGE, VIDEO, WEBCAM, YOUTUBE]

# Images config
IMAGES_DIR = ROOT / 'images'
DEFAULT_IMAGE = IMAGES_DIR / 'sn1.png'
DEFAULT_DETECT_IMAGE = IMAGES_DIR / 'sn1-detected.jpg'

# Videos config
VIDEO_DIR = ROOT / 'videos'
DEFAULT_VIDEO_PATH = VIDEO_DIR / 'SnakeDetection.mp4'
# ML Model config
MODEL_DIR = ROOT / 'weights'
SPECIES_MODEL = MODEL_DIR / 'bestspecies.pt'
SNAKE_MODEL = MODEL_DIR / 'best.pt'

# Webcam
WEBCAM_PATH = 0
