import multiprocessing
import time
import argparse
from typing import List
import psutil

def cpu_stress_worker(duration: int) -> None:
    """
    Worker function that stresses a single CPU core
    
    Args:
        duration: How long to run the stress test in seconds
    """
    start_time = time.time()
    while time.time() - start_time < duration:
        # Perform intensive floating point calculations
        _ = sum(pow(i * 2, 2) for i in range(10**5))

def monitor_cpu_usage(processes: List[multiprocessing.Process]) -> None:
    """
    Monitor and print CPU usage while stress test is running
    
    Args:
        processes: List of running worker processes to monitor
    """
    while any(p.is_alive() for p in processes):
        # Get CPU usage for each logical processor
        cpu_percent = psutil.cpu_percent(interval=1, percpu=True)
        print("\nCPU Usage per core:", end=" ")
        print(" ".join(f"{x:5.1f}%" for x in cpu_percent))
        
        # Check if any process has terminated
        for p in processes:
            if not p.is_alive():
                print(f"Process {p.pid} has terminated")

def run_stress_test(num_cores: int, duration: int) -> None:
    """
    Run a CPU stress test using specified number of cores
    
    Args:
        num_cores: Number of CPU cores to stress
        duration: Duration of stress test in seconds
    """
    # Create worker processes
    processes = []
    for _ in range(num_cores):
        p = multiprocessing.Process(target=cpu_stress_worker, args=(duration,))
        processes.append(p)
        p.start()
        
    print(f"\nStarted {num_cores} worker processes")
    print(f"Running stress test for {duration} seconds...")
    
    # Monitor CPU usage in separate thread
    monitor_cpu_usage(processes)
    
    # Wait for all processes to complete
    for p in processes:
        p.join()
    
    print("\nStress test completed")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CPU Stress Testing Tool')
    parser.add_argument('-n', '--num-cores', type=int, default=40,
                      help='Number of CPU cores to stress (default: all available cores)')
    parser.add_argument('-d', '--duration', type=int, default=60,
                      help='Duration of stress test in seconds (default: 60)')
    
    args = parser.parse_args()
    
    # If num_cores not specified, use all available cores
    if args.num_cores is None:
        args.num_cores = multiprocessing.cpu_count()
    
    print(f"\nSystem has {multiprocessing.cpu_count()} CPU cores available")
    print(f"Will stress {args.num_cores} cores for {args.duration} seconds")
    
    run_stress_test(args.num_cores, args.duration)