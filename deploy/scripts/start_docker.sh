#!/bin/bash
# Login to AWS ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 734359156280.dkr.ecr.us-east-1.amazonaws.com

# Pull the latest image
docker pull 734359156280.dkr.ecr.us-east-1.amazonaws.com/rrbecr:latest

# Check if the container 'campusx-app' is running
if [ "$(docker ps -q -f name=rrb-app)" ]; then
    # Stop the running container
    docker stop rrb-app
fi

# Check if the container 'campusx-app' exists (stopped or running)
if [ "$(docker ps -aq -f name=rrb-app)" ]; then
    # Remove the container if it exists
    docker rm campusx-app
fi

# Run a new container
docker run -d -p 80:8000 --name rrb-app 734359156280.dkr.ecr.us-east-1.amazonaws.com/rrbecr:latest