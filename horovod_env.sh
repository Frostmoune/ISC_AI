GCC_VERSION="5.5.0"
ENV_HOME="/GPUFS/sysu_hpcedu_302/ouyry/envs"

source ~/wyf/batch/env.sh
module load anaconda3/5.2.0
module load opencv/3.3.0
module load opt/gcc/$GCC_VERSION
source activate ouyry
export LD_LIBRARY_PATH="/GPUFS/app_GPU/anaconda3/5.2.0/lib:$LD_LIBRARY_PATH"
 
export CUDA_HOME="$ENV_HOME/cuda9_gcc$GCC_VERSION"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$CUDA_HOME/lib64/stubs:$LD_LIBRARY_PATH"
export LIBRARY_PATH="$CUDA_HOME/lib64:$CUDA_HOME/lib64/stubs:$LIBRARY_PATH"
 
export MPI_HOME="$ENV_HOME/openmpi4.0.0_gcc$GCC_VERSION"
export MPI_BIN="$MPI_HOME/bin"
export MPI_INCLUDE="$MPI_HOME/include"
export MPI_LIB="$MPI_HOME/lib"
export PATH="$MPI_BIN:$PATH"
export LD_LIBRARY_PATH="$MPI_LIB:$LD_LIBRARY_PATH"
 
export HOROVOD_CUDA_HOME="$CUDA_HOME"
export HOROVOD_NCCL_HOME="$ENV_HOME/nccl_gcc$GCC_VERSION/build"
export LD_LIBRARY_PATH="$ENV_HOME/lib:$LD_LIBRARY_PATH"