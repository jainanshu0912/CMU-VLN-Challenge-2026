FROM zhangjicmu/ubuntu24_ros:ai_module

# Install pip then PyTorch (CUDA 12.4) and GroundingDINO for GPU inference
USER root
RUN apt-get update && apt-get install -y --no-install-recommends python3-pip && \
    rm -rf /var/lib/apt/lists/* && \
    pip3 install --break-system-packages --no-cache-dir \
        --default-timeout=300 \
        torch torchvision \
        --index-url https://download.pytorch.org/whl/cu124 && \
    pip3 install --break-system-packages --no-cache-dir \
        --default-timeout=300 \
        --ignore-installed numpy \
        git+https://github.com/IDEA-Research/GroundingDINO.git
USER docker

# Copy updated dummy_vlm source into the existing workspace
COPY src/dummy_vlm /home/docker/ai_module/src/dummy_vlm

# Rebuild only dummy_vlm against the already-built workspace dependencies
WORKDIR /home/docker/ai_module
RUN /bin/bash -c "\
    source /opt/ros/jazzy/setup.bash && \
    colcon build \
      --cmake-args -DCMAKE_BUILD_TYPE=Release \
      --packages-select dummy_vlm"

# Ensure the updated install is sourced in every new shell
RUN echo "source /home/docker/ai_module/install/setup.bash" >> /home/docker/.bashrc
