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
# -p 5678:5678 for remote debugging
# -p 5678:5678 \
docker run --rm -it --ipc=host \
  -v "$(pwd)":/workspace -v /scratch/:/scratch/ -w /workspace \
  --user $(id -u):$(id -g) \
  --env XDG_CACHE_HOME --env WANDB_DATA_DIR --env WANDB_API_KEY --env NCCL_DEBUG \
  --gpus=\"device=${gpus}\" $image $command
