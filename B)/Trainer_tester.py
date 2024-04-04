import yaml
import subprocess
import os

def train(config):
    command = (
        f"python3 train.py --batch {config['batch_size']} "
        f"--epochs {config['epochs']} --data {config['data_train']} "
        f"--weights {config['weights']} --name {config['name']} "
        f"--freeze {config['freeze']} --device {config['device']}"
    )
    subprocess.run(command, shell=True)

def test(config, weights_path):
    command = (
        f"python3 test.py --data {config['data_test']} "
        f"--batch {config['batch_size']} --weights {weights_path} "
        f"--name {config['name']}"
    )
    subprocess.run(command, shell=True)

def main():
    with open("/home/aacastro/Alejandro/DQ_ACA_2024/B/configurations.yaml", "r") as file:
        configurations = yaml.safe_load(file)

    for config in configurations["configurations"]:
        print(f"Training model for configuration: {config['name']}")
        train(config)
        
        best_weights_path = f"runs/train/{config['name']}/weights/best.pt"
        
        print(f"Testing model for configuration: {config['name']}")
        test(config, best_weights_path)

if __name__ == "__main__":
    os.chdir("B/yolo_repo/yolov7/yolov7")
    main()
