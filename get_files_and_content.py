# Check for python files in the current directory
import os

files = os.listdir()
current_script = os.path.basename(__file__)  # gets "get_files_and_content.py"
count = 0

for file in files:
    if file.endswith(".py") and file != current_script:
        count += 1

# Open file.txt and write everything to it instead of printing
with open("file.txt", "w", encoding="utf-8") as outfile:
    outfile.write(f"There are {count} files which are:\n")
    for file in files:
        if file.endswith(".py") and file != current_script:
            outfile.write(file + "\n")
    
    outfile.write("\n\n")
    
    for file in files:
        if file.endswith(".py") and file != current_script:
            outfile.write(file + ":\n")
            # Write content of the file
            with open(file, "r", encoding="utf-8") as f:
                outfile.write(f.read())
                outfile.write("\n\n")