# Sourced by the job scripts: paths and Python env on the GPU server.
ROOT=/data/urvashib/openvla_panda
export HF_HOME=$ROOT/hf_cache
export PYTHONUNBUFFERED=1
source $ROOT/miniconda3/bin/activate openvla
cd $ROOT/openvla
