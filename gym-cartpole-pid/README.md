# The PID control implementation of cart-pole environment in gym

This is the PID implementation of cart-pole environment in gym.
I found that deep reinforcement learning is computational expensive, 
time-consuming and lack of robustness. 
Thus I spent some tome to finish this code to illustrate the effect of classical control algorithm.

In fact, as for both position loop and angle loop, the control method is PD control,
 as integral is useless in this model.
I also added a first-order low-pass filter to the position loop.

There are two bool variables in the programme, DIRECT_MAG and RANDOM_NOISE.

If DIRECT_MAG is true, the programme would switch to direct force mode, 
where control method could enforce consecutive force to the cart, 
instead of only 0/1 in original gym environment 
(which in that case, they represent giving 10N of force to the cart, left and right, respectively).

The modification of the simulation environment is in cart-pole_env.py.

If RANDOM_NOISE is true, the programme will add 
10N of force to the cart in next two sampling interval, as presenting the rapidity of the control algrithim.

I also left the reinforcement learning version of cart-pole, copied and modified from the code below:

<https://gist.github.com/n1try/2a6722407117e4d668921fce53845432#file-dqn_cartpole-py>

# Gym中直线倒立摆环境的PID控制

这是在Gym中直线倒立摆环境的PID算法实现。
我发现深度强化学习的算法既费力费时又鲁棒性不高，
因此我花了一点时间来完成这个算法以此显示传统控制算法的效果。

实质上对于位置、角度环的控制都为PD算法，积分控制在这个模型中没有必要。
我同时给位置环加了一阶低通滤波。

在程序中，有两个布尔变量，DIRECT_MAG与RANDOM_NOISE。

DIRECT_MAG为True时，则会切换到直接力模式，其中控制算法可以对小车施加连续大小的力，
而不是gym环境中原始的只有0/1两个动作空间（他们分别表示向左与向右施加10N的力）。

对于上述直接力的修改在cart-pole_env.py中。

RANDOM_NOISE为真时，则以百分之一的概率会有两个采样间隔中小车以10N的力右移。
这么做是为了展示控制算法的快速性。

我也留下了倒立摆环境的强化学习方法，作为对照，该程序来自：

<https://gist.github.com/n1try/2a6722407117e4d668921fce53845432#file-dqn_cartpole-py>


