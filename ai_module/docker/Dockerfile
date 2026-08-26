FROM zhangjicmu/ubuntu24_ros:ai_module

# Self-contained eval image: PyTorch, GroundingDINO, weights, BERT, sequential stack.
USER root
ENV DEBIAN_FRONTEND=noninteractive \
    HF_HOME=/home/docker/.cache/huggingface \
    TRANSFORMERS_CACHE=/home/docker/.cache/huggingface \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    PIP_DEFAULT_TIMEOUT=300

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3-pip wget ca-certificates git \
    && rm -rf /var/lib/apt/lists/*

# One CUDA 12.4 Torch only. Installing GroundingDINO *with* deps pulls a second
# CPU/CUDA-13 Torch and overflows the disk on --load.
RUN pip3 install --no-cache-dir \
        torch torchvision \
        --index-url https://download.pytorch.org/whl/cu124
RUN pip3 install --no-cache-dir --ignore-installed numpy \
        "setuptools>=30.3.0,<80" \
        "transformers>=4.36,<5" \
        pillow opencv-python-headless \
        timm supervision addict yapf pycocotools \
    && pip3 install --no-cache-dir \
        torch torchvision \
        --index-url https://download.pytorch.org/whl/cu124
RUN pip3 install --no-cache-dir --no-deps \
        git+https://github.com/IDEA-Research/GroundingDINO.git

# Config is in git. Weights are staged by docker/build.sh or downloaded here.
COPY docker/models /home/docker/models
RUN mkdir -p /home/docker/models /home/docker/.cache/huggingface \
    && if [ ! -s /home/docker/models/groundingdino_swint_ogc.pth ]; then \
         wget -O /home/docker/models/groundingdino_swint_ogc.pth \
           https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth; \
       fi \
    && test -s /home/docker/models/GroundingDINO_SwinT_OGC.py \
    && python3 -c "from transformers import AutoTokenizer, BertModel; \
AutoTokenizer.from_pretrained('bert-base-uncased'); \
BertModel.from_pretrained('bert-base-uncased')" \
    && chown -R docker:docker /home/docker/models /home/docker/.cache

COPY docker/fastdds_udp.xml /fastdds_udp.xml
ENV RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
    FASTRTPS_DEFAULT_PROFILES_FILE=/fastdds_udp.xml \
    RMW_FASTRTPS_USE_QOS_FROM_XML=1 \
    FASTDDS_BUILTIN_TRANSPORTS=UDPv4 \
    ROS_DOMAIN_ID=0 \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics

USER docker
WORKDIR /home/docker/ai_module

COPY --chown=docker:docker src/dummy_vlm /home/docker/ai_module/src/dummy_vlm
COPY --chown=docker:docker src/vlm_pipeline /home/docker/ai_module/src/vlm_pipeline
COPY --chown=docker:docker src/vlm_pipeline_live /home/docker/ai_module/src/vlm_pipeline_live
COPY --chown=docker:docker src/vlm_sequential /home/docker/ai_module/src/vlm_sequential

RUN /bin/bash -c "\
    source /opt/ros/jazzy/setup.bash && \
    colcon build \
      --cmake-args -DCMAKE_BUILD_TYPE=Release \
      --packages-select dummy_vlm vlm_pipeline vlm_pipeline_live vlm_sequential"

RUN echo "source /home/docker/ai_module/install/setup.bash" >> /home/docker/.bashrc \
    && echo "export HF_HOME=/home/docker/.cache/huggingface" >> /home/docker/.bashrc \
    && echo "export TRANSFORMERS_CACHE=/home/docker/.cache/huggingface" >> /home/docker/.bashrc \
    && echo "export RMW_IMPLEMENTATION=rmw_fastrtps_cpp" >> /home/docker/.bashrc \
    && echo "export FASTRTPS_DEFAULT_PROFILES_FILE=/fastdds_udp.xml" >> /home/docker/.bashrc \
    && echo "export RMW_FASTRTPS_USE_QOS_FROM_XML=1" >> /home/docker/.bashrc \
    && echo "export FASTDDS_BUILTIN_TRANSPORTS=UDPv4" >> /home/docker/.bashrc \
    && echo "export ROS_DOMAIN_ID=0" >> /home/docker/.bashrc
