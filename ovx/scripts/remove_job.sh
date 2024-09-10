#!/bin/bash -e

show_help() {
    echo ""
    echo "Usage: $0 <job_definition>"
    echo "  job_definition:    Name of the job definition to remove."
    echo ""
    echo "Environment variables required:"
    echo "  FARM_URL:          Omniverse Farm URL"
    echo "  FARM_API_KEY:      Omniverse Farm API Key"
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
check_environment_variable "FARM_API_KEY"

if [ "$#" -ne 1 ]; then
    echo "Error: Incorrect number of arguments. Expected 1, got $#."
    show_help
    exit 1
fi

# Using an undocumented API endpoint
curl -X "POST" "${FARM_URL}/queue/management/jobs/remove" \
    -H "Accept: application/json" \
    -H "Content-Type: application/json" \
    -H "X-API-KEY: ${FARM_API_KEY}" \
    -d '{
        "job_definition_name": "'"$1"'"
    }'
