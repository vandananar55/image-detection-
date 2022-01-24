# Image-Detection
This is an ML project using CNN to detect whether a person is wearing  a mask or not.

## Output Label : 
<b>masked<b> : 
  when a person is wearing mask
 <br>
<b>withoutMask<b> :
  when a person is not wearing mask

### Working Demo : 
  
![ezgif com-gif-maker](https://user-images.githubusercontent.com/94070137/150806810-623b2ab3-2d56-49db-8744-c9c1971b862c.gif)

## Project Description : 

This project consists of two parts : 

1. Training of Model
2. Prediction using trained model.

### Training of Model : 

For training of model,unzip train.zip and test.zip folders and give the path of both folders in facemask.py.

#### facemask.py :
This file is used to train the model. To run this file, execute the following command :

```
python facemask.py
```

After executing of facemask.py, we will get a trained model named mymodel.h5 which will be used in prediction part.

#### pred.py

This file will use the model trained in the previous step to get the output i.e. whether a person is wearing mask or not. To execute the code, run the following command.

```
python pred.py
```

After running this command, camera will start and you can see the output as a green lable showing Masked/without mask.
