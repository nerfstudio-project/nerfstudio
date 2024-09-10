#!/bin/bash -e

show_help() {
    echo ""
    echo "Usage: $0 <task_type> <task_args> <task_comment>"
    echo "  task_type:       Type of task to submit"
    echo "  task_args:       Arguments for the task"
    echo "  task_comment:    Comment or description for the task"
    echo ""
    echo "Environment variables required:"
    echo "  FARM_URL:        Omniverse Farm URL"
    echo "  FARM_USER:       Omniverse Farm user"
}

check_environment_variable() {
    local var_name=$1
    if [ -z "${!var_name}" ]; then
        echo "Error: Environment variable $var_name is not defined."
        show_help
        exit 1
    fi
}

check_environment_variable "FARM_URL"
check_environment_variable "FARM_USER"

if [ "$#" -ne 3 ]; then
    echo "Error: Incorrect number of arguments. Expected 3, got $#."
    show_help
    exit 1
fi

# echo '{
#         "user": "'"${FARM_USER}"'",
#         "task_type": "'"$1"'",
#         "task_args": {
#             "args": "'"$2"'"
#         },
#         "metadata": {
#             "_retry": {
#                 "is_retryable": false
#             }
#         },
#         "task_comment": "'"$3"'"
#     }'

# Reference: https://docs.omniverse.nvidia.com/farm/latest/farm_examples.html#integrating-the-job-with-omniverse-agent-and-queue
curl -X POST "${FARM_URL}/queue/management/tasks/submit" \
    -H "Accept: application/json" \
    -H "Content-Type: application/json" \
    -d '{
        "user": "'"${FARM_USER}"'",
        "task_type": "'"$1"'",
        "task_args": {
            "args": "'"$2"'"
        },
        "metadata": {
            "_retry": {
                "is_retryable": false
            }
        },
        "task_comment": "'"$3"'"
    }'
