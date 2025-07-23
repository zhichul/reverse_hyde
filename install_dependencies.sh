#!/bin/bash
set -eu

if [ ! -f "install_dependencies.sh" ]; then
    echo "Please run this script in the root folder of the repo."
    exit 1
fi

################## Setup ####################

# params
proj_root=$(pwd)
conda_prefix=$PSCRATCH/.conda_envs/reverse_hyde
install_verl=0
install_yourbench=0
install_doccot=0
install_uda=0
install_litsearch=1

# push some environment variables
echo "ROOT=$(pwd)" > .env

# get conda
source $(conda info --base)/etc/profile.d/conda.sh

################## get verl ####################
function getverl() {
    # get github repo
    if [ -e "lib/verl" ]; then
        echo "Verl already downloaded, skipping clone."
    else
        git clone https://github.com/zhichul/verl.git lib/verl
    fi
    cd lib/verl
    git checkout d3ec4a2e6b1482ed27e4cde9b005b5b8c661931b

    # make conda env
    conda create --prefix ${conda_prefix}/verl_v0.4.0 python==3.10 -y
    conda activate ${conda_prefix}/verl_v0.4.0
    export USE_MEGATRON=0 
    bash scripts/install_vllm_sglang_mcore.sh
    pip install --no-deps -e .
    
    # cleanup
    cd $proj_root
    conda deactivate
}
################## get yourbench ####################
function getyourbench() {
    # get github repo
    if [ -e "lib/yourbench" ]; then
        echo "yourbench already downloaded, skipping clone."
    else
        git clone https://github.com/zhichul/yourbench.git lib/yourbench
    fi
    cd lib/yourbench
    git checkout dd62da907df1b310989d9f77b60c4dced72f18bc

    # make conda env
    conda create --prefix ${conda_prefix}/yourbench_v0.3.1 python==3.12 -y
    conda activate ${conda_prefix}/yourbench_v0.3.1
    pip install -e .

    # cleanup
    cd $proj_root
    conda deactivate
}

################## get doc_cot ####################
function getdoccot() {
    # get github repo
    if [ -e "lib/doc_cot" ]; then
        echo "doc_cot already downloaded, skipping clone."
    else
        git clone https://github.com/zhichul/doc_cot.git lib/doc_cot
    fi
    cd lib/doc_cot
    # git checkout 
    
    # make conda env
    conda create --prefix ${conda_prefix}/doc_cot python==3.10 -y
    conda activate ${conda_prefix}/doc_cot
    pip3 install -r requirements.txt
    
    # get artifacts
    bash scripts/reassemble_bin.sh doc_cot/corpus/indices/olmo-mix-1124-pes2o-ids-to-file.parquet
    echo "\
    PES2O_PATH=/pscratch/sd/z/zlu39/olmo-mix-1124/data/pes2o/
    S2_API_KEY=klTlPNR9qxaTKnP604LdT6TRzThTv21M9JFCI8h1
    PROJECT_ROOT=$(pwd)
    " > .env

    # cleanup
    cd $proj_root
    conda deactivate
}

################## get UDA ####################
function getuda() {
    # get github repo
    if [ -e "lib/UDA-benchmark" ]; then
        echo "UDA-benchmark already downloaded, skipping clone."
    else
        git clone https://github.com/qinchuanhui/UDA-benchmark.git lib/UDA-benchmark
    fi
    cd lib/UDA-benchmark
    git checkout fca5237ac316e776d8dbccffa55ca29c0efdc185

    # make conda env
    conda create --prefix ${conda_prefix}/UDA-benchmark python==3.10 -y
    conda activate ${conda_prefix}/UDA-benchmark
    pip install -r requirements.txt

    # cleanup
    cd $proj_root
    conda deactivate
}

################## get UDA ####################
function getlitsearch() {
    # get github repo
    if [ -e "lib/LitSearch" ]; then
        echo "LitSearch already downloaded, skipping clone."
    else
        git clone https://github.com/princeton-nlp/LitSearch.git lib/LitSearch
    fi
    cd lib/LitSearch
    git checkout 42107c06b0ed0ed02ff7836ec285d3df2f0e5b09

    # make conda env
    conda create --prefix ${conda_prefix}/LitSearch python==3.10 -y
    conda activate ${conda_prefix}/LitSearch
    pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
    pip install numpy==1.23.5 transformers==4.37.2 datasets==4.0.0  sentence-transformers==2.2.2 InstructorEmbedding==1.0.1 rank-bm25==0.2.2 gritlm==1.0.0 openai==1.33.0

    # cleanup
    cd $proj_root
    conda deactivate
}

################## Main Pipeline ####################
if [ "$install_verl" -eq 1 ]; then
    getverl
fi
if [ "$install_yourbench" -eq 1 ]; then
    getyourbench
fi
if [ "$install_doccot" -eq 1 ]; then
    getdoccot
fi
if [ "$install_uda" -eq 1 ]; then
    getuda
fi
if [ "$install_litsearch" -eq 1 ]; then
    getlitsearch
fi

# ################### get data and model ####################
conda activate ${conda_prefix}/UDA-benchmark
huggingface-cli download GritLM/GritLM-7B
huggingface-cli download princeton-nlp/LitSearch --repo-type dataset
