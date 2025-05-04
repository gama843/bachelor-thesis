# bachelor-thesis

Vision-language models have made significant strides in interpreting and generating descriptions from visual data. However, their ability to perform complex relational reasoning remains a challenge. Relational reasoning involves understanding the relationships between different entities within a context, which is crucial for tasks such as visual question answering and scene understanding. Santoro et al. proposed a simple neural network module designed specifically to enhance relational reasoning in neural networks [1]. This thesis aims to explore the effectiveness of such modules when integrated into vision-language models. 

[1] Santoro, Adam et al. “A simple neural network module for relational reasoning.” Neural Information Processing Systems (2017). 


---

## Prerequisites & Setup

To install all required Python packages, run:

```bash
pip install -r requirements.txt

If you encounter the following error when running the code:

```bash
ImportError: libGL.so.1: cannot open shared object file: No such file or directory

Run:

```bash
sudo apt update
sudo apt install libgl1-mesa-glx

For reasonable running times, a CUDA-capable GPU is recommended (we tested on an NVIDIA P100). Ensure you have installed the appropriate NVIDIA drivers and CUDA toolkit.

Then, to ensure determinism in certain cuBLAS routines and support all functionality, set the following environment variable:

```bash
# config string that affects the determinism of certain cuBLAS routines
export CUBLAS_WORKSPACE_CONFIG=:4096:8

You will also need to set the OpenAI api key environment variable:

```bash
export OPENAI_API_KEY=""
