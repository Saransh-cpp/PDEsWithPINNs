import csv


def dat_to_csv(dat_file_name, csv_file_name, columns):

    with open(dat_file_name) as dat_file, open(
        csv_file_name, "w", newline=""
    ) as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(columns)
        for line in dat_file:
            if "#" in line:
                continue
            row = [field.strip() for field in line.split(" ")]
            csv_writer.writerow(row)
