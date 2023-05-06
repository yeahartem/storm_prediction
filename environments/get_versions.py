import ast
import pathlib
import subprocess
import sys

def main():
    file_path = pathlib.Path('environments/requirements.txt')
    file = open(file_path,'r')    
    for l in file.readlines():
        pkg_name = l.strip().split("==")[0]       
        command = [
            sys.executable,
            '-m',
            'pip',
            'show',
            pkg_name.strip(),
        ]
        command_output = subprocess.check_output(command).decode()
        for item in command_output.split("\n"):
            if "Version" in item:
                print(f'{pkg_name.strip()}=={item.strip().split(" ")[-1]}')


if __name__ == '__main__':
    main()