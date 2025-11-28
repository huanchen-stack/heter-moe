conda create -n tensorrtllm python=3.10 -y
conda activate tensorrtllm

# IMPORTANT
conda install -c conda-forge "python=3.10.13=hd12c33a_*"
# might need to export LD_LIBRARY_PATH whenever activate the env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# https://nvidia.github.io/TensorRT-LLM/installation/linux.html
pip install torch==2.9.0 torchvision --index-url https://download.pytorch.org/whl/cu130

# sudo apt-get -y install libopenmpi-dev
# Optional step: Only required for disagg-serving
# sudo apt-get -y install libzmq3-dev

pip install --upgrade pip setuptools && pip install tensorrt_llm
