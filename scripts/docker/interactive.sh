#!/bin/bash
# Usage: ./script <port_number>

if [ "$1" == "" ] ; then
  JUPYTER_PORT=8888
else
  JUPYTER_PORT=$1
fi

nvidia-docker run --name nvidia_vae_running -it --rm --ipc=host --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 --mount type=bind,source="$PWD",target="/code" -p $JUPYTER_PORT:8888 nvidia_vae

