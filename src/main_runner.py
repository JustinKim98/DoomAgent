import subprocess
import sys
import os


class ProccessRunner:
    def __init__(self, args, multiplayer=False):
        self.args = args
        self.multiplayer = multiplayer

    def run_process(self):
        run_two_processes_with_args(self.args, self.multiplayer)


def run_two_processes_with_args(args, multiplayer=False):
    # Define the commands for the two processes
    command1 = ["python", os.path.join(os.getcwd(), "src", "multi_host.py")] + args
    # Start the processes
    process1 = subprocess.Popen(command1)

    # Wait for them to finish
    if multiplayer:
        print("Invoking multiplayer")
        command2 = ["python", os.path.join(os.getcwd(), "src", "multi_join.py")] + args
        process2 = subprocess.Popen(command2)
        process2.wait()

    process1.wait()


if __name__ == "__main__":
    # Collect arguments passed to this script
    assert len(sys.argv) >= 2, "Enter an argument !"
    arguments = sys.argv[1:]  # Exclude the script name itself
    if arguments[0] not in ["corridor", "dtc", "deathmatch", "multi"]:
        raise AssertionError(
            "Please choose one of the four arguments 'corridor' 'dtc' 'deathmatch' or 'multi'"
            "and try again!"
        )

    run_two_processes_with_args(arguments, False)
