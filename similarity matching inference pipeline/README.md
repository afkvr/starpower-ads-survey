**Setup enviroment:**
```
conda create -n <enviroment_name> python=3.9 #or use -p with a specific path
conda activate <enviroment_name> 
pip install -r requirements.txt
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 #install torch with cuda independently 
```
**Data format** <br>
```bash
...\ROOT
+---refs_data
|   +---set1
|   |       set1_1.jpg
|   |       set1_2.jpg
|   |       set1_3.jpg
|   |       set1_4.jpg
|   |
|   +---set2
|   |       set2_1.jpg
|   |       set2_2.jpg
|   |       set2_3.jpg
|   |
|   \---set3
|    ...
\---test_data
    +---set1
    |       set1_10.jpg
    |       set1_12.jpg
    |       set1_31.jpg
    |       set1_44.jpg
    |
    +---set2
    |       set2_101.jpg
    |       set2_29.jpg
    |       set2_31.jpg
    |
    \---set3
     ...
```
Notice: <br>
&nbsp;&nbsp;&nbsp;&nbsp;Naming format is exactly like the format in similarity matching repo. <br>
&nbsp;&nbsp;&nbsp;&nbsp;Test data is optional and only required if you are evaluating the model. <br>
**Embedding model** <br>
&nbsp;&nbsp;&nbsp;&nbsp;Download the [model](https://drive.google.com/file/d/1Wxi7Mgm5jcakCYWlS0F9MtGkkS3okhTN/view?usp=drive_link). <br>
&nbsp;&nbsp;&nbsp;&nbsp;Create a checkpoints folder and put the model inside. <br>
**Inference pipeline** <br> 
&nbsp;&nbsp;&nbsp;&nbsp;First, a vector database must be created by running the create_vectordb script. <br>
&nbsp;&nbsp;&nbsp;&nbsp;The script has several arguments: <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;1. -r or --refs :Path to the reference images folder. <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;2. -p or --path_db :Path to store the database afterward, default is ./vectordb. <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;3. -e or --embedder :Path to the model checkpoint. <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;4. -s or --size :Size of the embedding vector, default is 128. <br>
&nbsp;&nbsp;&nbsp;&nbsp;Example: <br>
```
python create_vectordb.py -r .../ROOT/refs_data -e .../checkpoint/model.ckpt -s 128
```
&nbsp;&nbsp;&nbsp;&nbsp;The script return a vector database folder, the location is given by the path_db argument. <br>
&nbsp;&nbsp;&nbsp;&nbsp;The vector database folder contain an index.txt and a vectors.bin file. <br>
&nbsp;&nbsp;&nbsp;&nbsp;After having the database folder, the inference can be perform by using the "prediction" function in the inference.py file. <br> 
&nbsp;&nbsp;&nbsp;&nbsp;Custom inference pipeline can be easily build on top of this function. <br> 
&nbsp;&nbsp;&nbsp;&nbsp;For images that does not remotely look like any of the existing images in the database, the model will still return the images that it thinks look like the input the most. This can be overcome by using the distance matrix (D) and a fixed threshold. <br>   
**Evaluation pipeline** <br> 
&nbsp;&nbsp;&nbsp;&nbsp;First, create a vector database using the reference folder. <br>
&nbsp;&nbsp;&nbsp;&nbsp;After that, run the evaluation script. <br>
&nbsp;&nbsp;&nbsp;&nbsp;The script has several arguments: <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;1. -t or --test :Path to the test images folder. <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;2. -d or --database :Path to the vector database folder, default is ./vectordb. <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;3. -e or --embedder :Path to the model checkpoint. <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;4. -s or --size :Size of the embedding vector, default is 128. <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;4. -k or --k_neighbors :Size of the embedding vector, default is 3. <br>
&nbsp;&nbsp;&nbsp;&nbsp;Example: <br>
```
python evaluation.py -t ...\ROOT\test_data -e .../checkpoints/sep23.ckpt -d .../vectordb -k 3 -s 128
```
&nbsp;&nbsp;&nbsp;&nbsp;The script return top k accuracy on the test set given a reference vector database. <br>
