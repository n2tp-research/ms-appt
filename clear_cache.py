#!/usr/bin/env python3
"""
Clear the embedding cache to force re-computation with new model
"""

import os
import shutil
from pathlib import Path

def main():
    cache_dir = Path("embeddings_cache")
    
    if cache_dir.exists():
        print(f"Found cache directory: {cache_dir}")
        
        # List cache contents
        cache_files = list(cache_dir.glob("*"))
        if cache_files:
            print(f"Cache contains {len(cache_files)} files:")
            for f in cache_files:
                size_mb = f.stat().st_size / (1024 * 1024)
                print(f"  - {f.name} ({size_mb:.2f} MB)")
        
        # Ask for confirmation
        response = input("\nDo you want to clear the cache? This will force re-computation of embeddings. (y/N): ")
        
        if response.lower() == 'y':
            shutil.rmtree(cache_dir)
            print("Cache cleared successfully!")
            
            # Recreate empty directory
            cache_dir.mkdir(exist_ok=True)
            print("Created empty cache directory")
        else:
            print("Cache not cleared")
    else:
        print("No cache directory found")
        
if __name__ == "__main__":
    main()