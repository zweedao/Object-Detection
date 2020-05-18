# Scene Segmetation

Detecting a red barrel in different scenes using Gaussian model. 

Some examples:

![result 1](/Result/001.png)

![result 3](/Result/003.png)

![result 4](/Result/004.png)


## How to run:

### To use pre-trained model & detect red barrel in new images:
- Put images into folder "Test_Set"
- In Terminal, run "python detect_barrel.py"
- Result will be saved in folder "Result"

### If you want to annotate & train the model from scratch:

a) To annotate:
- In Terminal, run "python annotate_barrel.py"
- Draw polygon around the red barrel
- Annotation data will be saved in folder "Annotate"

b) To train model:
- In Terminal, run "python train.py"
- Trained models will be saved in folder "Model"

