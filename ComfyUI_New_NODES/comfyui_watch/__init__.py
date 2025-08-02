# Import the new WatchDetector class
from .watch_detector import WatchDetector
from .watchangle import WatchAngle
# Add the new node to the mapping.
# If you have other nodes, you can add this to your existing dictionary.
NODE_CLASS_MAPPINGS = {
    "WatchDetector": WatchDetector,
    "WatchAtAngleDetector": WatchAngle
}

# (Optional) You can also create a display name for your node.
NODE_DISPLAY_NAME_MAPPINGS = {
    "WatchDetector": "Watch Detector",
    "WatchAtAngleDetector":"Watch Angle Detector"
}

print("✅ Watch Detector custom node loaded.")
