FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

ENV TZ=Canada/Eastern
ENV DEBIAN_FRONTEND=noninteractive

#* For Pytorch builds
ENV TORCH_CUDA_ARCH_LIST="8.0 8.6 8.9 12.0+PTX" \
    TORCH_NVCC_FLAGS="-Xfatbin -compress-all" \
    FORCE_CUDA=1
#* last version should have +PTX


# Ensure CUDA toolkit path is visible for builds that might fall back to source
ENV CUDA_HOME=/usr/local/cuda \
    PATH=$CUDA_HOME/bin:$PATH \
    LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH


ARG USERNAME
ARG RENDER_GID
ARG VIDEO_GID
ARG USER_UID=1000
ARG USER_GID=$USER_UID


SHELL ["/bin/bash", "-c"]


#? System Packages
COPY apt_packages.txt .
RUN apt-get update && \
    xargs -a apt_packages.txt apt-get install -y --no-install-recommends && \
    rm apt_packages.txt && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    rm -rf /var/cache/apt/archives/*.deb

#? Install python dependencies
RUN pip3 install --upgrade pip
RUN pip3 install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128

COPY requirements.txt requirements.txt
COPY requirements_playground.txt requirements_playground.txt
RUN pip3 install -r requirements.txt
RUN pip3 install -r requirements_playground.txt
RUN pip3 install --no-build-isolation nvidia-cudnn-cu12==9.16.0.29


#? Download mujoco_menagerie
RUN python3 -c "from mujoco_playground import wrapper;from mujoco_playground import registry"


#? Create the user
RUN echo "Username is $USERNAME"
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

#? Create render group and add user to it
RUN groupadd --gid $RENDER_GID render \
    && usermod -aG render $USERNAME \
    && usermod -aG video $USERNAME

#? Add shortcut for jupyter notebook
RUN echo "alias jn='jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root'" >> /home/$USERNAME/.bashrc;


#? Set user and working directory
USER $USERNAME
WORKDIR /home/$USERNAME
