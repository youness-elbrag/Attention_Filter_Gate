#!/bin/bash 

echo "creating the folder ~/kaggale for testing the APi "
#after installing kaggale APi packages automatically will be created 
#folder into local directroy 
#/home/<username>/.kaggle
path_api="/home/yunus/.kaggle"
#creat APi kaggale account from kaggale 
api="kaggle.json"
#run commaned to download the datatset from kaggale 
cmd="kaggle datasets download -d adarshsng/heart-mri-image-dataset-left-atrial-segmentation"

mkcd()
{
    if [-f kaggle.json];then
         chmod 600 $api
    else
       echo "Ops the file kaggale does not exist ${/n}make you have API kaggale account"
    fi    
    #move file into locale directory 
    mv $api $path_api
    $cmd
    #unzip the file in project directory 
    unzip heart-mri-image-dataset-left-atrial-segmentation.zip 
    
}
mkcd

echo ".... Downloading Data is done"

