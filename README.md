# TextureNets_implementation

Implementation of the texture synthesis model in [Texture Networks: Feed-forward Synthesis of Textures and Stylized Images](https://arxiv.org/abs/1603.03417) of Ulyanov et al.

Based on Gatys' [code](https://github.com/leongatys/PytorchNeuralStyleTransfer)

## Training

The python script **train_g2d_periodic.py** trains a generator network.
The code requires the libraries: numpy, PIL and torch.
The VGG-19 perceptual loss between 2D images uses Gatys' implementation 
To run the code you need to get the pytorch VGG19-Model from the bethge lab by running:

sh download_models.sh 

Using [display](https://github.com/szym/display) is optional.

The name of the example texture is defined by the variable **input_name**.

The example textures go in the folder "Textures". 

The output file ***params.pytorch** contains the trained parameters of the generator network.
