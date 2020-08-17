.PHONY: build run

CONTAINER_NAME := ros-laptop2
VERSION := v1

build:
	docker build -t ${CONTAINER_NAME}:${VERSION} .

run:
	docker run -it --rm --net host --env DISPLAY=${DISPLAY} --volume /tmp:/transfer ${CONTAINER_NAME}:${VERSION} 
