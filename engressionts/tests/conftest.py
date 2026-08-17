import os

# Prevent xlstm import crash on Windows/environments without CUDA Toolkit during pytest collection
if "CUDA_HOME" not in os.environ and "CUDA_PATH" not in os.environ:
    os.environ["CUDA_HOME"] = os.path.dirname(os.path.abspath(__file__))
