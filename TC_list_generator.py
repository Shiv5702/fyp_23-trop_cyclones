import csv

input_file = "DataSources/ibtracs.NA.list.v04r00.csv"
output_file = "DataSources/trop_cyclones_lists.csv"

with open(input_file, "r") as f_in, open(output_file, "w", newline="") as f_out:
    reader = csv.DictReader(f_in)
    writer = csv.DictWriter(f_out, fieldnames=reader.fieldnames)
    writer.writeheader()
    
    for row in reader:
        if row["MLC_CLASS"] == "TS":
            writer.writerow(row)
