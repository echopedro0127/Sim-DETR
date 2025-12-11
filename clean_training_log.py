#!/usr/bin/env python3
"""
Clean training log by removing redundant progress bar updates.
Removes ANSI escape sequences and keeps only the final state of each progress bar.
"""
import re
import sys

def clean_log(input_file, output_file):
    """Remove redundant progress bar updates from training log."""

    with open(input_file, 'rb') as f:
        content = f.read()

    # Decode with error handling
    content = content.decode('utf-8', errors='ignore')

    # Remove ANSI escape sequences (like \033[A for cursor up)
    content = re.sub(r'\x1b\[[0-9;]*[A-Za-z]', '', content)

    # Split by lines, handling both \n and \r\n
    lines = content.split('\n')

    cleaned_lines = []

    for i, line in enumerate(lines):
        # Remove carriage returns and split by \r to get the last update
        if '\r' in line:
            # Split by \r and keep only the last part (final state)
            parts = line.split('\r')
            line = parts[-1]  # Keep only the last update

        # Skip empty lines that are just whitespace
        if line.strip() == '':
            continue

        # Check if this is a progress bar line
        is_progress = ('Training Iteration:' in line or
                      'compute st ed scores:' in line or
                      'Epoch:' in line and 'it/s' in line)

        if is_progress:
            # Check if the next line is also a progress line
            # If so, skip this one and keep the later one
            if i + 1 < len(lines):
                next_line = lines[i + 1]
                if '\r' in next_line:
                    next_line = next_line.split('\r')[-1]
                next_is_progress = ('Training Iteration:' in next_line or
                                   'compute st ed scores:' in next_line or
                                   'Epoch:' in next_line and 'it/s' in next_line)
                if next_is_progress:
                    continue  # Skip this line, keep the next one

        cleaned_lines.append(line)

    # Write cleaned log
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(cleaned_lines))

    # Print statistics
    import os
    original_size = os.path.getsize(input_file)
    cleaned_size = os.path.getsize(output_file)
    size_reduction = original_size - cleaned_size

    print(f"Original size: {original_size/1024:.1f} KB")
    print(f"Cleaned size: {cleaned_size/1024:.1f} KB")
    print(f"Size reduction: {size_reduction/1024:.1f} KB ({size_reduction/original_size*100:.1f}%)")
    print(f"\nCleaned log saved to: {output_file}")
    print(f"\nCleaning completed successfully!")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python clean_training_log.py <input_log> [output_log]")
        print("Example: python clean_training_log.py training_log_5epoch_fixed.txt training_log_clean.txt")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else input_file.replace('.txt', '_clean.txt')

    clean_log(input_file, output_file)
