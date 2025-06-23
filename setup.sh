
# install correct version of torch and torchvision according to your cuda version
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117    
pip install -r requirements.txt
conda install -y -c conda-forge dlib=19.24.2
pip install -e git+https://github.com/CompVis/taming-transformers.git@master#egg=taming-transformers
pip install -e git+https://github.com/openai/CLIP.git@main#egg=clip
pip install -e .
cd eval_tool/Deep3DFaceRecon_pytorch_edit
git clone https://github.com/NVlabs/nvdiffrast
cd ./nvdiffrast
pip install -e .
