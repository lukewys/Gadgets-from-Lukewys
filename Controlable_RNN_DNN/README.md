# Controlable CNN/RNN

This is a validation of controlable neural network. The task is given a MNIST picture vector and a conditional input (encoding in one-hot vector), outputs certain rotation of the image. We can see as long as the training data is large enough, DNN could accomplish the given rotation task in certain data distribution(in this case, MNIST). Note that here the neural network learnt a rotation process under MNIST data distribution instead of the vector rotation.

The same task and training could also apply to RNN, where in RNN the sequence is the slice of MNIST.

# 可控 RNN/DNN

这是对于神经网络可控的原理验证，定义控制为MNISI数据集的图像旋转，控制输入为独热向量。可以看到在训练数据足够多的情况下，DNN可以对于特定分布（MNIST数据集）完成给定的旋转任务。在这里神经网络学习到的不是矩阵旋转，而是在MNIST数据集的分布下的图像旋转。

同样的任务及训练被验证也可以施加在RNN上，在RNN中序列为MNIST的切片。