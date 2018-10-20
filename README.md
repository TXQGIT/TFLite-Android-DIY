# TFLite-Android-DIY
A simplified version of the TFLite demo


### Motivation:
In order to figure out how to use TensorFlow Lite on Android, 
I simplified the official TFLite demo and removed the complex camera module.
Instead, reading the input image by typing image name directly, 
then running the model inference and show the classification result.

### Environment:
Win10 + android-studio-ide-171.4443003-windows + sdk version 26

### Instruction:
1. Store the files (model, label and images) in models folder in a directory on your phone.
Change the value of the rootPath variable in the MainActivity.java file to the absolute path of this directory.
2. OK, it's done. Run it
