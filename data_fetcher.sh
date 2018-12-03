# to get COCO 2017 dataset from mirrors

mkdir data   # in case it is not present at root level 
mkdir data/coco/
mkdir data/coco/annotations
mkdir data/coco/images
mkdir data/coco/person_detection_results

wget http://msvocds.blob.core.windows.net/annotations-1-0-3/person_keypoints_trainval2017.zip
wget http://msvocds.blob.core.windows.net/coco2014/train2017.zip
wget http://msvocds.blob.core.windows.net/coco2014/val2017.zip

unzip person_keypoints_trainval2017.zip -d dataset/COCO/
unzip train2017.zip -d dataset/COCO/images
unzip val2017.zip -d dataset/COCO/images


rm -f person_keypoints_trainval2017.zip
rm -f train2017.zip
rm -f val2017.zip
