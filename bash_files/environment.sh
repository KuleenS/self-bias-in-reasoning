conda create -n self-bias python=3.12
conda activate self-bias
pip install vllm==0.11.2 --use-deprecated=legacy-resolver
pip uninstall numpy -y
pip install numpy==2.2.*
pip install pycountry pandas