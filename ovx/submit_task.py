import argparse
import os
import sys
import requests
from jinja2 import Environment, FileSystemLoader


def show_help():
    print("""
Usage: python script.py <task_type> <task_args> <task_comment>
  task_type:       Type of task to submit
  task_args:       Arguments for the task
  task_comment:    Comment or description for the task

Environment variables required:
  FARM_URL:        Omniverse Farm URL
  FARM_USER:       Omniverse Farm user
""")
    sys.exit(1)


def check_environment_variable(var_name):
    if var_name not in os.environ or not os.environ[var_name]:
        print(f"Error: Environment variable {var_name} is not defined.")
        show_help()


def submit_task(id, template_file):
    # Check required environment variables
    check_environment_variable("FARM_URL")
    check_environment_variable("FARM_USER")

    # Task arguments
    nucleus_hostname = 'nucleus.tpe1.local'
    task_type = "bash-gpu"
    task_args = f" \
        /run.sh \
        --download-src 'omniverse://{nucleus_hostname}/Projects/3DGS/datasets/{id}.zip' \
        --download-dest '/app/{id}.zip' \
        --upload-src '/app/{id}.mp4' \
        --upload-dest 'omniverse://{nucleus_hostname}/Projects/3DGS/ovx-test/{id}.mp4' \
        'python3 -m run --input /app/{id}.zip --output /app/{id}.mp4' \
    "
    task_comment = "ovx-test task"

    # Load the Jinja2 template
    env = Environment(loader=FileSystemLoader('.'))
    template = env.get_template(template_file)

    # Create the context for rendering the template
    context = {
        "FARM_USER": os.environ["FARM_USER"],
        "task_type": task_type,
        "task_args": task_args,
        "task_comment": task_comment
    }

    # Render the JSON payload
    json_payload = template.render(context)
    # print(json_payload)

    # Submit the task to the Omniverse Farm
    response = requests.post(
        f"{os.environ['FARM_URL']}/queue/management/tasks/submit",
        headers={"Accept": "application/json", "Content-Type": "application/json"},
        data=json_payload
    )

    # Check the response status
    if response.status_code == 200:
        print("Task submitted successfully!")
    else:
        print(f"Failed to submit task. Status code: {response.status_code}, Response: {response.text}")


def get_args():
     # Initialize the parser
    parser = argparse.ArgumentParser(description="Unzip a file in Python.")

    # Add arguments for input ZIP file and output directory
    parser.add_argument(
        '--id',
        type=str,
        required=True,
        help='ID of the ZIP file.'
    )
    parser.add_argument(
        '--template',
        type=str,
        default='ovx/task_definitions/bash_task.jinja',
        help='Path to the jinja file'
    )

    # Parse the arguments and return them
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    submit_task(args.id, args.template)

if __name__ == "__main__":
    main()