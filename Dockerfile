FROM osrf/ros:noetic-desktop-full

SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive
ENV KIRO_ROOT=/opt/kiro
ENV CATKIN_WS=/opt/kiro/catkin_ws
ENV PYTHONPATH=/opt/kiro/HybridSORT
ENV PIP_DEFAULT_TIMEOUT=300
ENV PIP_RETRIES=10
ARG TORCH_FLAVOR=cpu
ARG SKIP_MODEL_VALIDATE=0

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    libglib2.0-0 \
    libgl1 \
    libsm6 \
    libxext6 \
    libxrender1 \
    python3-dev \
    python3-pip \
    python3-setuptools \
    python3-wheel \
    ros-noetic-geometry-msgs \
    ros-noetic-message-filters \
    ros-noetic-sensor-msgs \
    ros-noetic-std-msgs \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel

RUN if [[ "${TORCH_FLAVOR}" == "cu118" ]]; then \
      python3 -m pip install --no-cache-dir \
        torch==2.1.2+cu118 \
        torchvision==0.16.2+cu118 \
        --index-url https://download.pytorch.org/whl/cu118; \
    else \
      python3 -m pip install --no-cache-dir \
        torch==2.1.2+cpu \
        torchvision==0.16.2+cpu \
        --index-url https://download.pytorch.org/whl/cpu; \
    fi

RUN python3 -m pip install --no-cache-dir \
    "numpy==1.24.4" \
    "opencv-python-headless==4.10.0.84" \
    "scipy==1.10.1" \
    "lap==0.5.12" \
    "loguru>=0.7,<1.0" \
    "matplotlib>=3.5,<4.0" \
    "PyYAML>=5.1" \
    "termcolor>=2.0" \
    "tqdm>=4.64,<5.0" \
    "py-cpuinfo>=9.0" \
    "yacs>=0.1.8,<1.0" \
    && python3 -m pip install --no-cache-dir --ignore-installed --no-deps \
    "psutil==5.9.8" \
    && python3 -m pip install --no-cache-dir --no-deps \
    "ultralytics-thop==2.0.19" \
    "ultralytics==8.3.40" \
    "filterpy==1.4.5"

COPY . /opt/kiro

RUN source /opt/ros/noetic/setup.bash \
    && SKIP_MODEL_VALIDATE="${SKIP_MODEL_VALIDATE}" python3 /opt/kiro/scripts/validate_runtime.py

RUN find /opt/kiro/catkin_ws/src -path '*/scripts/*.py' -type f -exec chmod +x {} \; \
    && chmod +x /opt/kiro/scripts/*.sh \
    && cd /opt/kiro/catkin_ws \
    && source /opt/ros/noetic/setup.bash \
    && catkin_make

WORKDIR /opt/kiro
ENTRYPOINT ["/opt/kiro/scripts/entrypoint.sh"]
CMD ["roslaunch", "human_tracking", "ros1_hybridsort_deploy.launch"]
