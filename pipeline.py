# 4_run_pipeline.py
import subprocess, sys
from pathlib import Path

def run(cmd):
    print(f"\n>>> {cmd}")
    if subprocess.run(cmd, shell=True).returncode != 0:
        print("FAILED!")
        sys.exit(1)

if __name__ == "__main__":
    run("python 1_prepare_data.py")
    run("python 2_train_siamese.py")
    test_folder = next(Path('train/samples').iterdir())
    run(f"python 3_infer_video.py \"{test_folder}\" yolov8n.pt models/siamese_final.pth")
    print("\nFULL PIPELINE DONE!")