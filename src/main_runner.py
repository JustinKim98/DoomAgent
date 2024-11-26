import subprocess
import sys



def run_two_processes_with_args(args):
    # Define the commands for the two processes
    command1 = ["python3", "multi_host.py"] + args
    command2 = ["python3", "multi_join.py"] + args

    # Start the processes
    process1 = subprocess.Popen(command1)
    process2 = subprocess.Popen(command2)

    # Wait for them to finish
    process1.wait()
    process2.wait()

if __name__ == "__main__":
    # Collect arguments passed to this script
    assert(len(sys.argv) >= 2), "Enter an argument !"
    arguments = sys.argv[1:]  # Exclude the script name itself
    if arguments[0] not in ["corridor", "dtc", "deathmatch"]:
        raise AssertionError("Please choose one of the three arguments 'corridor' 'dtc' 'deathmatch'"
                         "and try again!")

    run_two_processes_with_args(arguments)
