# Public, NVIDIA-maintained CUDA devel image: ships nvcc plus the full CUDA
# 12.6 toolkit for Ubuntu 24.04, matching this project's Cuda 12.xx / Nvidia
# GPU requirement without depending on any private/internal image.
FROM nvidia/cuda:12.6.3-devel-ubuntu24.04

ARG USERNAME=dev
ARG USER_UID=1000
ARG USER_GID=${USER_UID}

# Build tools + the only two libraries the project's tests link against
# (gtest, yaml-cpp), plus the shell/editor tooling the devcontainer (and its
# VS Code extensions) expect.
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
      cmake \
      git \
      openssh-client \
      ca-certificates \
      curl \
      wget \
      sudo \
      zsh \
      clang-format \
      clang-tidy \
      libgtest-dev \
      libyaml-cpp-dev \
      python-is-python3 \
      python3-pip \
      python3-matplotlib \
      python3-numpy \
      python3-tk \
    && rm -rf /var/lib/apt/lists/*

# The Ubuntu 24.04 base image already ships a default "ubuntu" user at
# uid/gid 1000; remove it so "dev" can take the same ids (needed for the
# bind-mounted workspace to have matching host/container permissions).
RUN existing_user="$(getent passwd "${USER_UID}" | cut -d: -f1)" \
    && if [ -n "$existing_user" ]; then userdel -r "$existing_user"; fi \
    && existing_group="$(getent group "${USER_GID}" | cut -d: -f1)" \
    && if [ -n "$existing_group" ]; then groupdel "$existing_group"; fi \
    && groupadd --gid ${USER_GID} ${USERNAME} \
    && useradd --uid ${USER_UID} --gid ${USER_GID} --create-home --shell /usr/bin/zsh ${USERNAME} \
    && echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/${USERNAME} \
    && chmod 0440 /etc/sudoers.d/${USERNAME}

USER ${USERNAME}
WORKDIR /home/${USERNAME}
