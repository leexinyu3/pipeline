#### Code explanation

######**train**
If you want to train the network using your own dataset from scratch, please run pipeline/train_net.py
The file pipeline/runs is the logger for training. The model you are training will be saved in pipeline/checkpoints

######demo.py#####
The pipeline/demo.py is used to show the process of inputting a photo which contains human's face and outputting the 
feature vector of the face.

###### **registration**

generate_ csv.py:  generate a CSV file. Each line contains the relative path of the image, and coordinates of 5 facial landmarks of  160*160 cropped faces (the size can be changed)

similarity.py  

Calculate the similarity of the two images according to the facial landmarks. when face poses is similar, the score is small, otherwise, the score is larger.

registration_face.py  

Select 5 or 6 images with very different poses  incuding one or two frontal faces and other images of different poses. by using similarity score.



#### How to use

Please put the HKPolyU database in to pipeline/data, which is organized as pipeline/data/1/meter_0.5/xxx.jpg

###### **1.registration**

Run registration_face.py

you will find there were 5 or 6 image in meter_0.5_selected

###### 2.test

**samedist_test.py**: same distance comparison

**crossdist_test.py** :cross distance comparison

**feature_dict.npy**  record  a dictionary with this structure. It contains the image path and the generated feature vector. eg: {image_path:feature vector} all images in new_data is in 112 * 112 size.

```
newdata
 --->1   
 	---> 0.5frontal      #store frontal faces of meter 0.5
 	---> 1.0frontal      #store frontal faces of meter 1.0
 	---> 1.5frontal      #store frontal faces of meter 1.5
 	---> meter_0.5       #store side faces of meter 0.5				
 	---> meter_1.0       #store side faces of meter 1.0
 	---> meter_1.5       #store side faces of meter 1.5 	
 --->2
 ...
 --->18
```

dir.txt stores the image path of the above structure.

**minidist_test:** test all the images in 1/1.5 folder by comparing them with 5 or 6 registered faces in 0.5 folder and choose the smallest Euclidean distance. 

**cropDataset160_featuredict.npy** record a dictionary with this structure. It contains the image path and the generated feature vector. eg: {image_path:feature vector}

```
cropDataset160
 --->1   
 	---> meter_0.5       #faces of meter 0.5				
 	---> meter_1.0       #faces of meter 1.0
 	---> meter_1.5       #faces of meter 1.5 	
 --->2
 ...
 --->18
```
NOTICE:
If you want to generate the               feature_dict.npy                 from 'newdata' by yourself, please run pipeline/test/samedist_crossdist/generate_dict.py
If you want to generate the         cropDataset160_featuredict     from 'cropDataset160' by yourself, please run a function called dict_feature() in minidist_test.py



