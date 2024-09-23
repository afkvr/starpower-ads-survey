**Setup enviroment:**
```
conda create -n <enviroment_name> python=3.9 #or use -p with a specific path
conda activate <enviroment_name> 
pip install -r requirements.txt
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 #install torch with cuda independently 
```
