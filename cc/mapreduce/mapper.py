import sys

# Input comes from standard input (stdin)
for line in sys.stdin:
    line = line.strip()  # remove leading/trailing whitespace
    words = line.split()  # split line into words
    for word in words:
        print(f"{word}\t1")  # output: word <tab> 1