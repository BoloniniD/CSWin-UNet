import re
import sys
import chardet

def detect_encoding(file_path):
    """Detect file encoding using chardet"""
    with open(file_path, 'rb') as f:
        rawdata = f.read(50000)  # Read first 50KB to detect encoding
    return chardet.detect(rawdata)['encoding']

def extract_ratios(log_file, output_file):
    # Try to detect file encoding
    try:
        encoding = detect_encoding(log_file)
        print(f"Detected encoding: {encoding}")
    except Exception as e:
        print(f"Encoding detection failed, using 'utf-8' with errors='replace': {e}")
        encoding = 'utf-8'

    # Regex pattern to match projection ratio lines
    pattern = r'Projection ratios - Min: ([\d.]+), Max: ([\d.]+), Mean: ([\d.]+)'

    # Store unique ratio entries in order of appearance
    unique_entries = set()
    entries_list = []
    line_count = 0
    extracted_count = 0

    try:
        with open(log_file, 'r', encoding=encoding, errors='replace') as f:
            for line in f:
                line_count += 1
                match = re.search(pattern, line)
                if match:
                    extracted_count += 1
                    min_val = match.group(1)
                    max_val = match.group(2)
                    mean_val = match.group(3)

                    # Create a tuple of the values
                    entry = (min_val, max_val, mean_val)

                    # Only add if we haven't seen this exact combination before
                    if entry not in unique_entries:
                        unique_entries.add(entry)
                        entries_list.append(entry)
    except Exception as e:
        print(f"Error processing file: {e}")
        return

    # Write to output file with index in original format
    with open(output_file, 'w', encoding='utf-8') as out:
        for idx, (min_val, max_val, mean_val) in enumerate(entries_list):
            # Recreate the original line format with index prefix
            out.write(f"{idx} Projection ratios - Min: {min_val}, Max: {max_val}, Mean: {mean_val}\n")

    print(f"Processed {line_count} lines")
    print(f"Found {extracted_count} ratio entries")
    print(f"Extracted {len(entries_list)} unique ratio entries to {output_file}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python extract_ratios.py input.log output.txt")
        sys.exit(1)

    extract_ratios(sys.argv[1], sys.argv[2])
