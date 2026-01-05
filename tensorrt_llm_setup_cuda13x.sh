# https://nvidia.github.io/TensorRT-LLM/installation/linux.html

onda create -n tensorrtllm python=3.12 -y
conda activate tensorrtllm

# MPI in the Slurm environment
conda remove -y mpi4py intel-mpi mpich openmpi
conda install -y -c conda-forge openmpi mpi4py

pip install torch==2.9.0 torchvision --index-url https://download.pytorch.org/whl/cu130

sudo apt-get -y install libopenmpi-dev
# Optional step: Only required for disagg-serving
sudo apt-get -y install libzmq3-dev

# Prevent pip from replacing existing PyTorch installation
CURRENT_TORCH_VERSION=$(python -c "import torch; print(torch.__version__)")
echo "torch==$CURRENT_TORCH_VERSION" > /tmp/torch-constraint.txt
pip install --upgrade pip setuptools && pip install tensorrt_llm -c /tmp/torch-constraint.txt

# When loading shared libraries (OpenMPI), look into conda's lib first!
# IMPORTANT!!!! BUT CONFLICT W/ GIT, SO USE SEPARATE SHELL FOR GIT
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
