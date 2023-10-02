import os
# Define the directories
output_dir = os.environ.get('OUTPUT_DIR', 'JL_basic')  # Default to 'JL_basic' if 'OUTPUT_DIR' is not set in the environment
class_dir = os.environ.get('CLASS_DIR', './woman')  # Default to './woman' if 'CLASS_DIR' is not set in the environment


print(output_dir)
print(class_dir)