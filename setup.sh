#!/usr/bin/env bash
set -e

# Script to start the flask docker container

if [ "$(sudo docker ps -q -a -f name=flask_container)" ]; then
	printf "Removing the flask container...\n\n"
        docker rm -f flask_container
	printf "\n"
	printf "Finished the removal of the container named flask_container\n"
fi
printf "=========================================================================================\n"
printf "Building flask image from Dockerfile...\n\n"
docker build . -t flask_image
printf "\n"
printf "Built a flask image named flask_image from the Dockerfile\n"
printf "=========================================================================================\n"
printf "Starting a container from the built image (port 80 and mount point to /files/pictures)\n\n"
docker run -d --name flask_container -v  /files/pictures:/files/pictures -p 80:80 flask_image
