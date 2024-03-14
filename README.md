# PubMedTempGraph
## Contributors: 
- [Daniel Bogacz](mailto:daniel.bogacz@stud.uni-heidelberg.de) (GitHub alias 'bgzdaniel')
- [Kenneth Styppa](mailto:kenneth.styppa@web.de) (GitHub alias 'KennyLoRI' and 'Kenneth Styppa')

**Remark**: Joint work is committed with both names in the commit message

## Overview

This project utilizes a combination of Langchain, ChromaDB, and llama.cpp to build a retrieval augmented generation system for medical question answering. The project is structured into modular pipelines that can be run end-2-end to first obtain the data, preprocess and embed the data, and later perform queries to interact with the retrieved information similar to a Q&A chatbot. Said chatbot contains two answer modes, a research mode and an overview mode. In the research mode, the user's question is answered using the latest research as context. In the overview mode a timeline overview of research relevant to the user's query is generated.

## Technologies Used

- **Langchain:** Langchain is a framework for developing applications powered by language models, including information retrievers, text generation pipelines and other wrappers to facilitate a seamless integration of LLM-related open-source software.

- **ChromaDB:** Chroma DB is an open-source vector storage system designed for efficiently storing and retrieving vector embeddings.

- **llama.cpp:** llama.cpp implements Meta's LLaMa2 architecture in efficient C/C++ to enable a fast local runtime.

## Installation & set-up

1. **Prerequisites:**
   - Ensure you have Python installed on your system. Your Python version should match 3.10.
   - Ensure to have conda installed on your system.
   - Create a folder where you want to store the project.

2. **Create a Conda Environment:**
   - Create a conda environment
   - Activate the environment
   ```bash
   conda create --name your_project_env python=3.10
   conda activate your_project_env
   ```

4. **Clone the Repository into your working directory:**
   ```bash
   git clone git@github.com:bgzdaniel/PubMedTempGraph.git
   ```
   When using Mac set pgk_config path:
   ```bash
   export PKG_CONFIG_PATH="/opt/homebrew/opt/openblas/lib/pkgconfig"
   ```

   then switch to the working directory of the project:
   ```bash
   cd PubMedTempGraph
   ```
   
6. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   
7. **Llama.cpp GPU installation:**
   (When using CPU only, skip this step.)

   This part might be slightly tricky, depending on which system the installation is done. We do NOT recommend installation on Windows. It has been tested, but requires multiple components which need to be downloaded. Please contact [Daniel Bogacz](mailto:daniel.bogacz@stud.uni-heidelberg.de) for details.

   **Linux:**
   ```bash
   CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install --upgrade --force-reinstall llama-cpp-python --no-cache-dir
   ```

   **MacOS:**
   ```bash
   CMAKE_ARGS="-DLLAMA_METAL=on" FORCE_CMAKE=1 pip install --upgrade --force-reinstall llama-cpp-python --no-cache-dir
   ```

If anything goes wrong in this step, please contact [Daniel Bogacz](mailto:daniel.bogacz@stud.uni-heidelberg.de) for **Linux** installation issues and [Kenneth Styppa](mailto:kenneth.styppa@web.de) for **MacOS** installation issues. Also refer to the installation guide provided [here](https://python.langchain.com/docs/integrations/llms/llamacpp) and also [here](https://llama-cpp-python.readthedocs.io/en/latest/install/macos/)

6. **Download chroma store and model files and place them into the right location:**
   - Go to [this](https://drive.google.com/drive/folders/1-6FxGDDKGD-sMwT2Pax7VVMLzuZUH0DG) Google drive link and download the ChromaDB store (folder called `chroma_store_abstracts`) as well as the [model files](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/blob/main/mistral-7b-instruct-v0.2.Q6_K.gguf).
   - Insert the ChromaDB store at `data/chroma_store`
   - Insert the model file at `data/mistral-7b-instruct-v0.2.Q6_K`

## Usage
### Using the Q&A system
1. **Navigate to the main folder `PubMedTempGraph` in your terminal:**
   ```bash
   cd PubMedTempGraph
   ```

2. **Activate the Q&A System:**
   ```bash
   python -m chat
   ```
3. **Interact with the system:**
   - Ask your question
   ```bash
   Please enter your question: [your_question]
   ```
   - Ask another question (and so on)
  
Note: Running the system for the first time might take some additional seconds because the model and the vector database has to be initialized.
  
### Trouble-shooting: 
- If you encounter an issue during your usage install pyspellchecker separately and try again:
  ```bash
  pip install pyspellchecker
  ```
- When encountering issues in the Llama.cpp installation, make sure you have NVIDIA Toolkit installed. Check with:
   ```bash
   nvcc --version
   ```
   Something similar to the following should appear:
   ```bash
   nvcc: NVIDIA (R) Cuda compiler driver
   Copyright (c) 2005-2023 NVIDIA Corporation
   Built on Wed_Feb__8_05:53:42_Coordinated_Universal_Time_2023
   Cuda compilation tools, release 12.1, V12.1.66
   Build cuda_12.1.r12.1/compiler.32415258_0
   ```
   Also make sure that CMake is installed on your system.

### Optional usage possibilities 

1. **Embedding of abstracts or paragraphs:**
   - For embedding abstracts files like `studies_{year}.csv` are required. Place them in `data/studies`. See [here](https://drive.google.com/drive/folders/139ZXhBJ3WVpzhiY7fzl9K2mx84VmKXmB?usp=sharing) for the data. For example for the file `studies_2019.csv` the following command needs to be executed:
   ```bash
   python -m abstract2chroma.abstract2vec --year "2019"
   ```
   
2. **Loading embeddings to the vector database ChromaDB:**
   - For loading abstract-based embeddings to the vector database, the files of type `embeddings_{year}.csv` are required. Place it in `data/embeddings`. Create them with the step **Embedding of abstracts or paragraphs**. For example for the files `embeddings_2019.csv` and `embeddings_2020.csv` the following command needs to be executed:
   ```bash
   python -m abstract2chroma.vec2chroma --years "2019,2020" --create_new
   ```
   The argument `--create_new` instructs the script to delete an existing database and create a new one. If a database already exists at `data/chroma_store`, then leave the argument parameter out of the execution command.
   
5. **Running Validation and Evaluation:**
   - For the evaluation, BleuRT is required. First clone bleuRT:
   ```bash
   git clone https://github.com/google-research/bleurt.git
   ```
   Go in into the subfolder 'bleurt':
   ```bash
   cd bleurt
   ```
   Specifically for *MacOS*: Because `tensorflow` is differently named under MacOS, the install requirements have to be changed. Go to `bleurt/setup.py` and change in the list variable `install_requires` the entry `tensorflow` to `tensorflow-macos`. It should look like the following:
   ```python
   install_requires = [
    "pandas", "numpy", "scipy", "tensorflow-macos", "tf-slim>=1.1", "sentencepiece"
   ]
   ```

   Save the file.

   Install bleuRT with the following:
   
   ```bash
   pip install . 
   ```

   The used BleuRT model can be found [here](https://storage.googleapis.com/bleurt-oss-21/BLEURT-20-D12.zip). Place it under `data/BLEURT-20-D12`.
   
   For evaluation the steps **Embedding of abstracts or paragraphs** and **Loading embeddings to the vector database ChromaDB** have to be completed. Run the evaluation with the following command:
   ```bash
   python -m eval
   ```
