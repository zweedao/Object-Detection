# How to run:

1) To use pre-trained model & detect red barrel in new images:
- Put images into folder "Test_Set"
- In Terminal, run "python detect_barrel.py"
- Result will be saved in folder "Result"


2) If you want to annotate & train the model yourself, do this:
a) To annotate:
- In Terminal, run "python annotate_barrel.py"
- Draw polygon around the red barrel
- Annotation data will be saved in folder "Annotate"

b) To train model:
- In Terminal, run "python train.py"
- Trained models will be saved in folder "Model"
