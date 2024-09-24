**Setup enviroment:**
```
conda create -n <enviroment_name> python=3.9 #or use -p with a specific path
conda activate <enviroment_name> 
pip install -r requirements.txt
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 #install torch with cuda independently 
```
**Data format:** <br>
```bash
\ROOT
+---set1
|       set1_1.jpg
|       set1_2.jpg
|       set1_3.jpg
|       set1_4.jpg
|
+---set2
|       set2_1.jpg
|       set2_2.jpg
|       set2_3.jpg
|
\---set3
...
```
Notice: 
 &nbsp;&nbsp;&nbsp;&nbsp;Set's name shouldn't contain any " _ " symbol. <br>
 &nbsp;&nbsp;&nbsp;&nbsp;Images of set "a" must has the naming format: seta_index.ext. <br>
 &nbsp;&nbsp;&nbsp;&nbsp;Images names shouldn't contain more than one " _ " symbol. <br>
**Data augmentation:** <br>
To augment data in the dataset, run the augmentation.py script in the dataset folder. <br>
The data augmentation consist of 5 transformation: rotation by +5 and -5 degree, Gaussian noise injection and two color adjustments transformation.
```
cd dataset 
python augmentation.py #change the "root" variable in line 37 to your dataset's root first.
```
