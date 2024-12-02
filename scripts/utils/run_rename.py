import os
import shutil
import sys

def main(src_base, old_postfix, new_postfix):
    base = os.path.dirname(src_base)
    pid = os.path.basename(src_base)
    old_name = f"{pid}_sub_{old_postfix}.nii.gz"
    new_name = f"{pid}_sub_{new_postfix}.nii.gz"
    old_file = os.path.join(src_base, old_name)
    assert os.path.exists(old_file), f"{old_file} does not exist."
    new_file = os.path.join(src_base, new_name)
    assert not os.path.exists(new_file), f"{new_file} already exists."
    os.rename(old_file, new_file)
    print(f"renamed from {old_name} to {new_name}")

if __name__ == '__main__':
    if len(sys.argv) == 4:
        src_base = str(sys.argv[1])
        old_postfix = str(sys.argv[2])
        new_postfix = str(sys.argv[3])
        main(src_base, old_postfix, new_postfix)
    else:
        print("Usage: python script.py <src_base_path> <old_postfix> <new_postfix>")