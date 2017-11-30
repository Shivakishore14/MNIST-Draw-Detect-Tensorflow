# MNIST-Draw-Detect-Tensorflow

Tensorflow classification of MNIST dataset trained on

1. Regression model
2. Convolution model
3. Perceptron model
4. Recurrent neural network model

> Each model is trained simultaniously for only 20000 steps with batch-size of 50

[https://draw-detect-mnist-tensorflow.herokuapp.com/](https://draw-detect-mnist-tensorflow.herokuapp.com/)

This project gives insight on the training dataset used [MNIST Heatmap](https://github.com/Shivakishore14/MNIST-heatmap-tensorflow)

### How to run
```
$ pip install -r requirements.txt
$ gunicorn app:app
```

This project is heavily inspired by the [tensorflow-mnist](https://github.com/sugyan/tensorflow-mnist).
