#!/bin/bash

# Default values
# image="konstantinjdobler/tight-budget:main"
image="konstantinjdobler/tight-budget:hindsight"
command="bash"
gpus="none"

# Function to parse the command line arguments
parse_arguments() {
  local in_command=false

  while [[ $# -gt 0 ]]; do
    case "$1" in
    -g)
      shift
      gpus="$1"
      ;;
    -i)
      shift
      image="$1"
      ;;
    *)
      if [ "$in_command" = false ]; then
        command="$1"
      else
        command="${command} $1"

      fi
      in_command=true
      ;;
    esac
    shift
  done
}

# Call the function to parse arguments
parse_arguments "$@"

# Rest of your script
echo "image: $image"
echo "command: $command"
echo "gpus: $gpus"

# Look for WANDB_API_KEY
if [ -z "$WANDB_API_KEY" ]; then
  export WANDB_API_KEY=$(awk '/api.wandb.ai/{getline; getline; print $2}' ~/.netrc)
  if [ -z "$WANDB_API_KEY" ]; then
    echo "WANDB_API_KEY not found"
  else
    echo "WANDB_API_KEY found in ~/.netrc"
  fi
else
  echo "WANDB_API_KEY found in environment"
fi

# NOTE: --ipc=host for full RAM and CPU access or -m 300G --cpus 32 to control access to RAM and cpus
# -p 5678:5678 \
# using 0:0 because of rootless docker, which is mapped to the original user outside of the container

docker run --rm -it --ipc=host \
  -v "$(pwd)":/workspace -v /raid/konstantin.dobler/:/raid/konstantin.dobler/ -w /workspace \
  --user 0:0 \
  --env XDG_CACHE_HOME --env WANDB_DATA_DIR --env WANDB_API_KEY --env CUDA_LAUNCH_BLOCKING \
  --gpus=\"device=${gpus}\" $image $command
