# vm's default python3 version is 3.10


################################
# INSTALL NV TOOLKIT [REQUIRED]
################################

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/13.0.0/local_installers/cuda-repo-ubuntu2204-13-0-local_13.0.0-580.65.06-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2204-13-0-local_13.0.0-580.65.06-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2204-13-0-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda-toolkit-13-0

export PATH=/usr/local/cuda-13.0/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:$LD_LIBRARY_PATH
nvcc --version

################################

# trt build: https://nvidia.github.io/TensorRT-LLM/latest/installation/build-from-source-linux.html# 

# TensorRT LLM uses git-lfs, which needs to be installed in advance.
apt-get update && apt-get -y install git git-lfs
git lfs install

git clone https://github.com/NVIDIA/TensorRT-LLM.git
cd TensorRT-LLM
git submodule update --init --recursive
git lfs pull

# build 

pip install cmake
apt install libnuma-dev
apt install tensorrt-dev tensorrt-libs
apt install openmpi-bin libopenmpi-dev   #?
apt install libucx-dev   #?
apt install ninja-build

# rm -rf cpp/build

python3 ./scripts/build_wheel.py --extra-cmake-vars ENABLE_MULTI_DEVICE=0 --extra-cmake-vars WARNING_IS_ERROR=ON --extra-cmake-vars ENABLE_UCX=0 --micro_benchmarks
pip install -e .
# clean build: python3 ./scripts/build_wheel.py --clean

pip install -r requirements.txt
python -c "import tensorrt_llm"


# ===================================================
#  official docker image (PREFERED)
# ---------------------------------------------------
#    ALSO NEED TO INSTALL CUDA TOOLKIT FIRST!!!
#    https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tensorrt-llm/containers/devel?version=1.2.0rc6.post3
#    With the top-level directory of the TensorRT LLM repository cloned to your local machine, you can run the following command to start the development container:
#    potentially need: python scripts/build_wheel.py --help 2>&1 | grep -iE "nixl|ucx"

################# PREFERED!!! ####################
pip install cmake
pip install -r requirements.txt

apt-get install libnuma-dev libnccl2 libnccl-dev openmpi-bin libopenmpi-dev libnvinfer-dev libnvonnxparsers-dev libucx-dev ninja-build -y

rm -rf cpp/build
make -C docker ngc-devel_run LOCAL_USER=1 DOCKER_PULL=1 IMAGE_TAG=1.3.0rc2
./scripts/build_wheel.py --clean --use_ccache --cuda_architectures=native

python -c "import tensorrt_llm"
################# PREFERED!!! ####################

# OR
docker run --rm -it --ipc=host --ulimit memlock=-1 --ulimit stack=67108864  \
           --gpus=all \
           --env "CCACHE_DIR=/code/tensorrt_llm/cpp/.ccache" \
           --env "CCACHE_BASEDIR=/code/tensorrt_llm" \
           --env "CONAN_HOME=/code/tensorrt_llm/cpp/.conan" \
           --workdir /code/tensorrt_llm \
           --tmpfs /tmp:exec \
           --volume .:/code/tensorrt_llm \
           nvcr.io/nvidia/tensorrt-llm/devel:1.3.0rc2
./scripts/build_wheel.py --clean --use_ccache --cuda_architectures=native
