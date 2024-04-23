import subprocess
import concurrent.futures
import os

def run_command_on_two_gpus(command):
    """
    Run the command on both GPUs by setting CUDA_VISIBLE_DEVICES to "0,1".
    """
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0,1"
    subprocess.run(command, shell=True, env=env)

def run_commands_in_batches(commands, batch_size=9):
    """
    Run a list of commands in batches, allowing up to batch_size commands to run in parallel.
    Each command will run on both GPUs.
    """
    # Split commands into batches
    for i in range(0, len(commands), batch_size):
        batch = commands[i:i+batch_size]
        with concurrent.futures.ThreadPoolExecutor(max_workers=batch_size) as executor:
            futures = [executor.submit(run_command_on_two_gpus, cmd) for cmd in batch]
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"Command raised an exception: {e}")
        # Optionally, add a delay or a condition here if you need to ensure GPUs cool down or for other reasons

# Load commands from file
commandsList = []
with open('run_experiments_0.9_25_25_6_6.sh', 'r') as file:
    for line in file:
        commandsList.append(line.strip())

# Run commands in batches
run_commands_in_batches(commandsList, batch_size=8)
