from MainController import runme
import sys
import os

if __name__ == "__main__":
    config_filename = sys.argv[1]
    output_path = sys.argv[2]
    if sys.argv[3]:
        output_path = os.path.join(output_path, str(sys.argv[3]))
        os.mkdir(output_path)
    runme(config_filename, output_path)