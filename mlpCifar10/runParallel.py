import subprocess
import concurrent.futures
import os

def run_command_on_gpu(command, gpu_id):
    """
    Run the command on the specified GPU by setting CUDA_VISIBLE_DEVICES.
    """
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    subprocess.run(command, shell=True, env=env)

def run_commands_in_batches(commands, batch_size=9):
    """
    Run a list of commands in batches, allowing up to batch_size commands to run in parallel.
    Each command will run on the specified GPU.
    """
    # Split commands into batches
    for i in range(0, len(commands), batch_size):
        batch = commands[i:i+batch_size]
        with concurrent.futures.ProcessPoolExecutor(max_workers=batch_size) as executor:
            futures = []
            for idx, cmd in enumerate(batch):
                gpu_id = idx % 2  # Toggle between GPU 0 and GPU 1
                futures.append(executor.submit(run_command_on_gpu, cmd, gpu_id))
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"Command raised an exception: {e}")

# Load commands from file
commandsList = []
with open('run_experiments_1Diag_1.sh', 'r') as file:
    for line in file:
        commandsList.append(line.strip())

# Run commands in batches
run_commands_in_batches(commandsList, batch_size=10)
