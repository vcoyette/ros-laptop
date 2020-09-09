.PHONY: build run

CONTAINER_NAME := ros-laptop2
VERSION := v1
IP := $(shell ip route get 1.2.3.4 | awk '{print $$7}')


build:
	docker build -t ${CONTAINER_NAME}:${VERSION} .

run:
	docker run -it --rm \
		--net host \
		--env DISPLAY=${DISPLAY} \
		--env ROS_IP=${IP} \
		--env ROS_MASTER_IP=${duckie_ip} \
		--volume /tmp:/transfer \
		${CONTAINER_NAME}:${VERSION}
