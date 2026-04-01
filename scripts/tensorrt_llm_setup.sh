conda create -n tensorrtllm python=3.10 -y
conda activate tensorrtllm

# IMPORTANT
conda install -c conda-forge "python=3.10.13=hd12c33a_*"
conda install -c conda-forge mpi4py

# https://nvidia.github.io/TensorRT-LLM/installation/linux.html
pip install torch==2.9.0 torchvision --index-url https://download.pytorch.org/whl/cu130

# sudo apt-get -y install libopenmpi-dev
# Optional step: Only required for disagg-serving
# sudo apt-get -y install libzmq3-dev

pip install --upgrade pip setuptools && pip install tensorrt_llm

pip install parameterized pytest

# IMPORTANT!!!! BUT CONFLICT W/ GIT, SO USE SEPARATE SHELL FOR GIT
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# ===================================================================
# ========================== cuda130 ================================
# ===================================================================
# https://nvidia.github.io/TensorRT-LLM/installation/linux.html 

# conda create -n tensorrtllm python=3.10 -y
# conda activate tensorrtllm

conda install -c conda-forge mpi4py -y

pip install uv
uv pip install torch==2.9.0 torchvision --index-url https://download.pytorch.org/whl/cu130
sudo apt-get -y install libopenmpi-dev
CURRENT_TORCH_VERSION=$(python -c "import torch; print(torch.__version__)")
echo "torch==$CURRENT_TORCH_VERSION" > /tmp/torch-constraint.txt
# pip install --extra-index-url https://pypi.nvidia.com/ "tensorrt-llm==1.1.0" -c /tmp/torch-constraint.txt
aria2c -x 16 -s 16 "https://pypi.nvidia.com/tensorrt-llm/tensorrt_llm-1.1.0-cp312-cp312-linux_x86_64.whl" -d /tmp/
pip install /tmp/tensorrt_llm-1.1.0-cp312-cp312-linux_x86_64.whl --extra-index-url https://pypi.nvidia.com/ -c /tmp/torch-constraint.txt

export PMIX_MCA_psec=native
export PRTE_MCA_plm_ssh_agent=""
export PRTE_ALLOW_RUN_AS_ROOT=1
export PRTE_ALLOW_RUN_AS_ROOT_CONFIRM=1
