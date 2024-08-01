import random

def generate_command():
    experiment_name = "diffDiag"
    expNum = random.randint(1, 1000)
    
    commands = []
    for k in range(2, 31):
        sparsity = 1 - k / 3072
        num_layers = 1
        diagPos = sorted(random.sample(range(3072), k))
        
        command = (f"CUDA_VISIBLE_DEVICES=1 python train_cifar10.py "
                   f"--expName {experiment_name} "
                   f"--net mlpmixer "
                   f"--n_epochs 300 "
                   f"--lr 1e-3 "
                   f"--expNum {expNum} "
                   f"--sparsity {sparsity} "
                   f"--num_layers {num_layers} "
                   f"--diagPos {' '.join(map(str, diagPos))}")
        
        commands.append(command)
    
    with open("commands.sh", "w") as file:
        for command in commands:
            file.write(command + "\n")

if __name__ == "__main__":
    generate_command()
