import sys
import os


def load_modules():
    # Get the path of the current file (confmodules.py)
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Navigate to the 'src' directory (assumes 'src' is in the server folder one level up)
    src_path = os.path.join(current_dir, '..', 'src')

    # Add the 'src' directory to the system path if it's not already there
    if src_path not in sys.path:
        sys.path.append(src_path)

    # Now you can import app and coding modules
    try:
        import agent
        import core
        import env
        import option_learning
        import plot
        import subgoal_discovery
        import trajectory_sampler
        import utils
        print("Modules loaded successfully.")
    except ImportError as e:
        print(f"Error importing modules: {e}")
