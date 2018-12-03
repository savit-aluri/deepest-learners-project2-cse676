Group : Deepest Learners
Savit Aluri - savitvar@buffalo.edu
Harshal Anil Patil - harshala@buffalo.edu
Avinash Kommineni - akommine@buffalo.edu

***** RUNNING THE PROJECT *******

Directory Structure:
.
├── data
│   └── coco
├── experiments
│   ├── coco
│   └── mpii
├── lib
│   ├── core
│   ├── dataset
│   ├── Makefile
│   ├── models
│   ├── nms
│   └── utils
├── models
│   └── pytorch
├── output
│   └── coco
├── pose_estimation
│   ├── _init_paths.py
│   ├── output
│   ├── train_densenet.py
│   ├── train.py
│   └── valid.py
├── README.md
└── running_main.ipynb

21 directories, 8 files

Required libraries: 
EasyDict==1.7
opencv-python==3.4.1.15
Cython
scipy
pandas
pyyaml
json_tricks
scikit-image
tensorboardX>=1.2
torchvision

Data Acquisition and Preprocessing: 
To acquire the data, we fetch the data from the official COCO documentation. In our code, to fetch the script, simply run
sh data_fetcher.sh
This fetches the data and organizes the data into structures.
 

Training: 
Then we just get to the root folder and run:
python pose_estimation/train.py --cfg experiments/coco/resnet50/256x256_d256x3_adam_lr1e-3.yaml

The configuration for the network can be found at network/

The weights are learnt and can be found in network/weights/ as vgg19-dcbb9e9d.pth

In addition, since we couldn't run all the 200 epochs as suggested by the author, we used the trained model which can be found at:

https://onedrive.live.com/?authkey=%21AKqtqKs162Z5W7g&id=56B9F9C97F261712%2110693&cid=56B9F9C97F261712

Validation: 
```
python pose_estimation/valid.py \
    --cfg experiments/coco/resnet50/256x192_d256x3_adam_lr1e-3.yaml \
    --flip-test \
    --model-file models/pytorch/pose_coco/pose_resnet_50_256x192.pth.tar


```
Runs the evaluation score and returns the Average Precision score. 


Link to the original code: 
https://github.com/Microsoft/human-pose-estimation.pytorch
