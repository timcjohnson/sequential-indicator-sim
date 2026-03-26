#generate indicator simulations for the 
import sequential_indicator_sim as sis
import numpy as np
from pathlib import Path
from glob import glob


def convert_npz_to_txt(directory: str = ".") -> None:
    """
    Search for .npz files in directory and convert them to .txt format.
    
    Each .npz file is expected to contain x, y, z, and category arrays.
    Output .txt files will contain one point per line: x y z category
    """
    npz_files = glob(f"{directory}/*.npz")
    
    if not npz_files:
        print(f"No .npz files found in {directory}")
        return
    
    for npz_path in sorted(npz_files):
        try:
            # Load the .npz file
            data = np.load(npz_path)
            x = data['x']
            y = data['y']
            z = data['z']
            category = data['category']
            
            # Generate output filename by replacing .npz with .txt
            txt_path = npz_path.replace('.npz', '.txt')
            
            # Write to .txt file
            with open(txt_path, 'w') as f:
                for xi, yi, zi, cat in zip(x, y, z, category):
                    f.write(f"{xi} {yi} {zi} {int(cat)}\n")
            
            print(f"Converted {npz_path} -> {txt_path} ({len(x)} points)")
        except Exception as e:
            print(f"Error converting {npz_path}: {e}")


exit_code = sis.main([
"--known", "known_indicators.txt",
"--output", "gift1_isim",
"--x-start", "0", "--x-end", "1.9304", "--x-num", "52",
"--y-start", "0", "--y-end", "0.7366", "--y-num", "28",
"--z-start", "0", "--z-end", "0.9652", "--z-num", "40",
"--categories", "1,2,3",
"--vrange-x", "5.", "--vrange-y", "5.0", "--vrange-z", "0.1",
"--seed", "43",
"--search-radius-x", ".5",
"--search-radius-y", ".5",
"--search-radius-z", "0.2",
"--interp-x-num", "104",
"--interp-y-num", "56",
"--interp-z-num", "80",
"--num-realizations", "20",
"--num-cores", "20",
])

convert_npz_to_txt()
