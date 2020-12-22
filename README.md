# Tianchi2019_Imagenet_Attack

No.9 Solution for [Tianchi2019 ImageNet Attack Challenge](https://tianchi.aliyun.com/competition/entrance/231761/introduction)  
Generate adversarial images to attack ImageNet classification models  
Team: niceA  

## Environment  
python 2.7.0 tensorFlow1.8.0 

Pretrained tensorflow models can be found [here](https://github.com/tensorflow/models/tree/master/research/adv_imagenet_models)

## Run
> sh tianchi.sh [ your imagepath ] [ your output imagepath ] [ your epsilon ] 

Remember we delete the title of dev.csv in folder images 
