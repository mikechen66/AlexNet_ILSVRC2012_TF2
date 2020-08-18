
It includes the auxilary materials for role playing on the ILSVRC2012. Some of the material are duplicate. However, it is useful for users to get to know the usage of ILSVRC images and files. Becuase it lacks of comprehensive code descripiton of the ILSVRC 2012 Championship in both Google and Github. Users are hard to know what has happened with the ISLVRC2012. Most of the develpers aopted a quite small dataset to conduct the role play on the AlexNet. In contrast, very few developers has a deep dive on the AlexNet with so huge dataset(160 GB). Therefore, user can take the following information for reference.  

ilsvrc_untar.sh:for the first-order file(s). 
ilsvrc_classes_untar.sh for the second-order files(1000 tars related to 1000 classes) included in 
ILSVRC2012_img_train.tar. 

ILSVRC2012_img_train.tar (about 138 GB)
ILSVRC2012_img_val.tar (about 6.3 GB)
ILSVRC2012_img_test.tar (about 13.7 GB)
Please make sure the three tar files in your current directory and use the CD command to operate the files. 

#  train/
#  ├── n01440764
#  │   ├── n01440764_10026.JPEG
#  │   ├── n01440764_10027.JPEG
#  │   ├── ......
#  ├── ......
#  val/
#  ├── n01440764
#  │   ├── ILSVRC2012_val_00000293.JPEG
#  │   ├── ILSVRC2012_val_00002138.JPEG
#  │   ├── ......
#  ├── ......
