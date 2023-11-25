cd third_party/GPT4RoI/

pip install --upgrade pip  # enable PEP 660 support
pip install setuptools_scm
pip install --no-cache-dir  -e .
# please use conda re-install the torch, pip may loss some runtime lib
# conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia 

pip install ninja
pip install flash-attn --no-build-isolation

cd mmcv-1.4.7
MMCV_WITH_OPS=1 pip install -e .
cd ../../../

pip install -r requirements.txt
echo "export PATH=\$PATH:\$HOME/.local/bin" >> ~/.bashrc
export PATH=$PATH:$HOME/.local/bin

# NOTE: dependency hell.
pip install pydantic==1.10.10

# Remove the line 1271 in transformers/generation/utils.py for GPT4ROI
sed -i '1271d' $(pip show transformers | grep Location | cut -d ' ' -f2)/transformers/generation/utils.py

# Add wandb api
# ref: https://docs.wandb.ai/guides/track/environment-variables
# MY_WANDB_API_KEY='5622beac064f35ae9bec9d910a4d20b0191c015b'
MY_WANDB_API_KEY='9524f85f305826397e257653098a70b69c97f3cc'
export WANDB_API_KEY=$MY_WANDB_API_KEY
echo "export WANDB_API_KEY=$MY_WANDB_API_KEY" >> ~/.bashrc

# Show full error trace from hydra
echo "export HYDRA_FULL_ERROR=1" >> ~/.bashrc


nvidia-smi

# Download azcopy
TMP_DIR=tmp/
AZCOPY_URL=https://aka.ms/downloadazcopy-v10-linux
AZCOPY_TAR_FILE="$TMP_DIR/azcopy-v10-linux.tar.gz"
AZCOPY_FILE="$TMP_DIR/azcopy"

"$AZCOPY_FILE" --version
has_azcopy=$?

if [[ has_azcopy -eq 0 ]]; then
    echo "azcopy exists"
else
    echo "azcopy does not exist"
    mkdir -p $TMP_DIR
    wget $AZCOPY_URL -O $AZCOPY_TAR_FILE
    file_to_be_extracted="$(tar -tvf $AZCOPY_TAR_FILE | grep -E 'azcopy$' | awk '{print $6}')"
    tar -zxvf $AZCOPY_TAR_FILE  -C "$TMP_DIR" "$file_to_be_extracted"
    mv $TMP_DIR/$file_to_be_extracted $TMP_DIR
    rm $AZCOPY_TAR_FILE 
    rmdir "$(dirname $TMP_DIR/$file_to_be_extracted)"
    chmod 777 $AZCOPY_FILE
    export PATH=$PATH:$(pwd)/$TMP_DIR
    echo "export PATH=\$PATH:$(pwd)/$TMP_DIR" >> ~/.bashrc
fi

# Show full error trace from hydra
echo "export HYDRA_FULL_ERROR=1" >> ~/.bashrc

# Change dataset to hg download
TARGET_DATASETS_VER="2.13.1"
version="$(pip show datasets | grep Version | awk '{print $2}')"
if [[ $version == $TARGET_DATASETS_VER ]]; then
    echo "datasets version is $TARGET_DATASETS_VER, changing it to use azcopy..."
    pip_package_path="$(pip show datasets | grep Location | awk '{print $2}')"
    download_file_path="$pip_package_path/datasets/utils/file_utils.py"
    if [[ -f $download_file_path.bak ]]; then
        cp $download_file_path.bak $download_file_path
    fi
    cp $download_file_path $download_file_path.bak
    sed -i '609 i\
            # NOTE(xiaoke): An intrusion to use azcopy to download from Azure blob storage\
            elif "blob.core.windows.net" in url:\
                process_id = -1\
                try:\
                    import torch\
                    if torch.distributed.is_initialized():\
                        process_id = torch.distributed.get_rank()\
                except ImportError:\
                    logger.warning("no torch found, cannot determine whether is in ddp mode")\
                except RuntimeError:\
                    logger.warning("torch.distributed is not initialized, cannot determine whether is in ddp mode")\
\
                logger.warning(f"[process {process_id}] Try to use azcopy to download from Azure blob storage")\
                import subprocess\
\
                has_azcopy = subprocess.run(["azcopy"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode\
                if has_azcopy != 0:\
                    logger.warning(f"[process {process_id}] azcopy not found, using http_get, which is slow")\
                    http_get(\
                        url,\
                        temp_file,\
                        proxies=proxies,\
                        resume_size=resume_size,\
                        headers=headers,\
                        cookies=cookies,\
                        max_retries=max_retries,\
                        desc=download_desc,\
                    )\
                else:\
                    logger.warning(f"[process {process_id}] azcopy found, using azcopy")\
                    result = subprocess.run(\
                        ["azcopy", "cp", url, temp_file.name],\
                    )\
                    if result.returncode != 0:\
                        raise ConnectionError(\
                            f"azcopy failed with return code {result.returncode}"\
                        )\
' $download_file_path
else
    echo "datasets version is NOT $TARGET_DATASETS_VER, not changed"
fi

# For debug
sudo apt-get update
if [[ $? -ne 0 ]]; then
    apt-get update
fi
sudo apt-get install -y tmux htop vim lsof
if [[ $? -ne 0 ]]; then
    apt-get install -y tmux htop vim lsof
fi

# Tmux config
curl -L https://raw.githubusercontent.com/hamvocke/dotfiles/master/tmux/.tmux.conf -o - >> ~/.tmux.conf

# Vim config
# Install vim-plug
curl -fLo ~/.vim/autoload/plug.vim --create-dirs \
    https://raw.githubusercontent.com/junegunn/vim-plug/master/plug.vim

cat << EOF > ~/.vimrc
set tabstop=4
set shiftwidth=4
set expandtab
set smartindent
set nu
set hlsearch
set ignorecase
set mouse=a

call plug#begin()
Plug 'tpope/vim-surround'
Plug 'tpope/vim-commentary'
Plug 'davidhalter/jedi-vim'
call plug#end()

let g:jedi#force_py_version = 3 " Force using Python 3
EOF
vim +'PlugInstall --sync' +qa

# Install gpustat
pip install gpustat

# echo pwd
echo "pwd: $(pwd)"