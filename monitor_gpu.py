#!/usr/bin/env python3
"""Monitor GPU utilization during training."""

import subprocess
import time
import argparse
from datetime import datetime


def get_gpu_stats():
    """Get GPU utilization stats using nvidia-smi."""
    try:
        result = subprocess.run([
            'nvidia-smi', 
            '--query-gpu=timestamp,gpu_name,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,power.limit',
            '--format=csv,noheader,nounits'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            stats = result.stdout.strip().split(', ')
            return {
                'timestamp': stats[0],
                'gpu_name': stats[1],
                'temperature': float(stats[2]),
                'gpu_util': float(stats[3]),
                'mem_util': float(stats[4]),
                'mem_used': float(stats[5]),
                'mem_total': float(stats[6]),
                'power_draw': float(stats[7]),
                'power_limit': float(stats[8])
            }
    except Exception as e:
        print(f"Error getting GPU stats: {e}")
    return None


def main():
    parser = argparse.ArgumentParser(description='Monitor GPU utilization')
    parser.add_argument('--interval', type=int, default=5, help='Monitoring interval in seconds')
    parser.add_argument('--output', type=str, default='gpu_stats.log', help='Output log file')
    args = parser.parse_args()
    
    print(f"Monitoring GPU every {args.interval} seconds. Press Ctrl+C to stop.")
    print(f"Logging to: {args.output}")
    
    with open(args.output, 'w') as f:
        f.write("Timestamp,GPU,Temp(C),GPU_Util(%),Mem_Util(%),Mem_Used(MB),Mem_Total(MB),Power(W),Power_Limit(W)\n")
        
        try:
            while True:
                stats = get_gpu_stats()
                if stats:
                    line = f"{datetime.now()},{stats['gpu_name']},{stats['temperature']:.1f},"
                    line += f"{stats['gpu_util']:.1f},{stats['mem_util']:.1f},"
                    line += f"{stats['mem_used']:.0f},{stats['mem_total']:.0f},"
                    line += f"{stats['power_draw']:.1f},{stats['power_limit']:.1f}"
                    
                    print(f"\rGPU: {stats['gpu_util']:5.1f}% | Mem: {stats['mem_used']:5.0f}/{stats['mem_total']:5.0f}MB ({stats['mem_util']:4.1f}%) | Power: {stats['power_draw']:5.1f}W | Temp: {stats['temperature']:3.0f}C", end='')
                    
                    f.write(line + "\n")
                    f.flush()
                
                time.sleep(args.interval)
                
        except KeyboardInterrupt:
            print("\n\nMonitoring stopped.")


if __name__ == '__main__':
    main()