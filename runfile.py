from MainController import runme
import sys

if __name__ == "__main__":
    config_filename = sys.argv[1]
    output_path = sys.argv[2]
    runme(config_filename, output_path)