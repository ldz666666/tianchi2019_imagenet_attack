# tianchi2019_imagenet_attack

No 10 Solution for [Tianchi2019 ImageNet Attack Challenge](https://tianchi.aliyun.com/competition/entrance/231761/introduction)  
Team: niceA  

## Envirionment  
Python 2.7.0 Tensorflow1.8.0 

pertrained tensorflow models can be found [here](https://github.com/tensorflow/models/tree/master/research/adv_imagenet_models)

## Run
> sh tianchi.sh [ your imagepath ] [ your output imagepath ] [ your epsilon ] 

Remember we delete the first row (title) of dev.csv in folder images 
