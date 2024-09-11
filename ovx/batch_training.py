import csv
import os
from ovx.submit_task import submit_task


def main():
    # Define the path to your CSV file
    csv_file = "test_src/roomplan_dataset.csv"
    template_file = "ovx/task_definitions/bash_task.jinja"
    # Loop through the CSV file and execute the command for each file
    with open(csv_file, mode='r', newline='', encoding='utf-8') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            id = row["File name"]
            print(id)
            # submit_task(id, template_file)


if __name__ == '__main__':
    main()