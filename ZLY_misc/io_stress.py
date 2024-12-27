import multiprocessing
import numpy as np
import time
import argparse
import psutil
from typing import Tuple

def create_large_arrays(size_mb: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create source and destination arrays of specified size
    """
    # Calculate array size (1 MB = 1024 * 1024 bytes)
    array_size = (size_mb * 1024 * 1024) // 8  # 使用8字节的float64
    return np.random.random(array_size), np.zeros(array_size)

def memory_stress_worker(size_mb: int, duration: int) -> None:
    """
    Worker function that performs memory intensive operations
    模拟内存到寄存器的频繁数据移动
    """
    # 创建源数组和目标数组
    source_array, dest_array = create_large_arrays(size_mb)
    start_time = time.time()
    
    while time.time() - start_time < duration:
        # 强制数据在内存和寄存器之间移动
        # 使用步长1确保连续访问，但编译器不能轻易优化
        for i in range(0, len(source_array), 1):
            dest_array[i] = source_array[i] * 1.0  # 强制运算以使用寄存器
            
        # 防止编译器过度优化
        dest_array[0] = source_array[-1]

def monitor_system(processes, size_mb: int) -> None:
    """
    Monitor CPU and memory usage
    """
    while any(p.is_alive() for p in processes):
        # 获取CPU和内存使用情况
        cpu_percent = psutil.cpu_percent(interval=1, percpu=True)
        memory = psutil.virtual_memory()
        
        print("\nSystem Status:")
        print("CPU Usage per core:", " ".join(f"{x:5.1f}%" for x in cpu_percent))
        print(f"Memory Usage: {memory.percent}% (Used: {memory.used/1024/1024:.0f}MB)")
        
        # 检查进程状态
        for p in processes:
            if not p.is_alive():
                print(f"Process {p.pid} has terminated")

def run_memory_stress(num_cores: int, size_mb: int, duration: int) -> None:
    """
    Run memory stress test using specified number of cores
    """
    processes = []
    total_memory_needed = size_mb * num_cores
    
    # 检查是否有足够的内存
    available_memory = psutil.virtual_memory().available / (1024 * 1024)  # Convert to MB
    if total_memory_needed > available_memory * 0.8:  # 留出20%的安全余量
        print(f"Warning: Test might use more memory ({total_memory_needed}MB) than safely available ({available_memory:.0f}MB)")
        return

    # 创建工作进程
    for _ in range(num_cores):
        p = multiprocessing.Process(
            target=memory_stress_worker,
            args=(size_mb, duration)
        )
        processes.append(p)
        p.start()
        
    print(f"\nStarted {num_cores} worker processes")
    print(f"Each process using {size_mb}MB memory")
    print(f"Running for {duration} seconds...")
    
    # 监控系统状态
    monitor_system(processes, size_mb)
    
    # 等待所有进程完成
    for p in processes:
        p.join()
    
    print("\nMemory stress test completed")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Memory-to-Register Stress Testing Tool')
    parser.add_argument('-n', '--num-cores', type=int, default=36,
                      help='Number of CPU cores to use')
    parser.add_argument('-s', '--size', type=int, default=1024*2,
                      help='Memory size in MB per process (default: 500)')
    parser.add_argument('-d', '--duration', type=int, default=60,
                      help='Duration in seconds (default: 60)')
    
    args = parser.parse_args()
    
    if args.num_cores is None:
        args.num_cores = multiprocessing.cpu_count()
    
    print(f"\nSystem has {multiprocessing.cpu_count()} CPU cores available")
    print(f"Will use {args.num_cores} cores")
    print(f"Memory per process: {args.size}MB")
    print(f"Duration: {args.duration} seconds")
    
    run_memory_stress(args.num_cores, args.size, args.duration)