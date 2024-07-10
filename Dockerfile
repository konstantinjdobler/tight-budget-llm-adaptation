# syntax=docker/dockerfile:1

FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV RUNTIME docker
ENV DEBIAN_FRONTEND=noninteractive

# install some basics, micro for having an actually good terminal-based editor, curl for getting uv 
# software-properties-common necessary for add-apt-repository
RUN apt-get update && apt-get install -y git gcc g++ nano openssh-client curl software-properties-common && \
    rm -rf /var/lib/apt/lists/* && apt-get clean

# install python 3.11; the -dev is important for fasttext, else complains about missing Python.h
RUN add-apt-repository ppa:deadsnakes/ppa && apt-get update && \ 
    apt-get -y install python3.11-dev && \
    rm -rf /var/lib/apt/lists/* && apt-get clean

# Create a non-privileged user
# See https://docs.docker.com/develop/develop-images/dockerfile_best-practices/#user
ARG UID=10001
ARG USERHOME="/home/nlpresearcher"
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home $USERHOME \
    --shell "/sbin/nologin" \
    --uid "${UID}" \
    nlpresearcher

# make important dirs with correct permissions for non-priviliged user
# use chmod -777 instead of chown s.t. docker run with different UID also works
RUN mkdir /opt/venv && chmod -R 777 /opt/venv/ && \
    mkdir /opt/reqs && chmod -R 777 /opt/reqs/ && \
    mkdir -p $USERHOME/.cache/uv && chmod -R 777 $USERHOME/.cache/uv

# good to copy these instead of mounting, that way the requirements can be easily reconstructed just from having the image 
COPY ./requirements.lock /opt/reqs/requirements.lock
COPY ./pyproject.toml /opt/reqs/pyproject.toml
COPY ./.python-version /opt/reqs/.python-version

# Switch to non-root user now to prevent issues when running w/ non-root uid later on
USER nlpresearcher
WORKDIR $USERHOME
# Install uv
ADD --chmod=755 https://astral.sh/uv/install.sh ./install.sh
RUN ./install.sh && rm -rf ./install.sh 
# ...and add uv to PATH
ENV PATH="${USERHOME}/.cargo/bin:${PATH}"

# create venv and set VIRTUAL_ENV to tell uv about it
RUN uv venv -p 3.11 /opt/venv 
ENV VIRTUAL_ENV=/opt/venv

# CC / CXX env vars needed for latest fasttext
ENV CC="/usr/bin/gcc"
ENV CXX="/usr/bin/g++"

# Download dependencies as a seperate step to take advantage of Docker's caching.
# Leverage a cache mount to $USERHOME/.cache/uv to speed up subsequent builds.
RUN --mount=type=cache,target=$USERHOME/.cache/uv,uid=$UID uv pip install -r /opt/reqs/requirements.lock

RUN --mount=type=cache,target=$USERHOME/.cache/uv,uid=$UID MAX_JOBS=20 uv pip install --no-build-isolation 'flash-attn>=2.5' 

# cache doesn't work here
RUN --mount=type=cache,target=$USERHOME/.cache/uv,uid=$UID \
    mkdir $USERHOME/flash-attn-for-kernel && git clone https://github.com/Dao-AILab/flash-attention.git $USERHOME/flash-attn-for-kernel && \
    MAX_JOBS=20 uv pip install --no-build-isolation $USERHOME/flash-attn-for-kernel/csrc/layer_norm/ && \
    MAX_JOBS=20 uv pip install --no-build-isolation $USERHOME/flash-attn-for-kernel/csrc/rotary/ && \
    MAX_JOBS=20 uv pip install --no-build-isolation $USERHOME/flash-attn-for-kernel/csrc/xentropy/ && \
    rm -rf $USERHOME/flash-attn-for-kernel

# ``activate'' venv automatically when using docker run
ENV PATH="/opt/venv/bin:${PATH}"

# set this as default entrypoint workdir, code should be mounted there
WORKDIR /workspace
