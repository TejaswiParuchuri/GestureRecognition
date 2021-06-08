# GestureRecognition
Predict the gestures in a video using trainded model based on the key points generated for the video using Deep Learning with the help of Convolution Neural Networks based on Tensor flow

### Key points generation for vidoes:
For videos first images were generated (30 images approx for 1 second length of video) and from these images key points were generated in csv for each video using the instruction steps and code provided in https://github.com/prashanthnetizen/posenet_nodejs_setup repository

### Gesture Prediction:
  * Four Models (using K-Nearest Neighbors, Decision Tree, Random Forest, Neural Networks) were trained using the gesture videos (70:30 training and testing split) 
  * These trained models are used to predict gestures and were deployed on Google App Engine
