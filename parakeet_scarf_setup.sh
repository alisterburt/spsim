# drop into a gpu node on SCARF
module load GPUmodules
conda create --name parakeet python=3.9
conda activate parakeet

# install some specific dependencies
conda install -c conda-forge fftw gxx=9 cudatoolkit-dev pytest pytest-cov

# env variables for compilation
export CXX=$(which g++)
export CUDACXX=$(which nvcc)
export CMAKE_CUDA_ARCHITECTURES=37 # CUDA arch for k80s on SCARF
export FFTW_ROOT=$CONDA_PREFIX

# install and test...
git clone https://github.com/rosalindfranklininstitute/amplus-digital-twin.git
pushd amplus-digital-twin
git submodule update --init --recursive
pip install -r requirements.txt
pip install -e .
pytest
popd