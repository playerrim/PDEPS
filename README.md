# 1.论文介绍

## 1.1 背景介绍

​偏微分方程（PDEs）在物理问题（如声波传播、流体流动、电磁场等）中有重要应用。传统的数值方法（如计算流体力学和气象预报）已经成功应用于实际问题，但随着问题规模和复杂性的增加，计算复杂度变得非常高。同时，随着数值方法在工业和学术界的广泛使用，仿真数据的积累也在增加。数据驱动的深度学习方法被提出以减轻计算负担，一旦训练好神经网络，预测过程非常快。然而，数据获取成本高，且当标签数据不足时，模型的泛化能力差。

近年来，基于物理信息的方法被提出，这些方法利用深度神经网络的通用近似能力和自动微分技术。与传统的PDE求解器相比，深度学习求解器不仅可以处理参数化仿真，还可以解决传统求解器无法处理的逆问题或数据同化问题。物理信息神经网络（PINNs）方法通过保留控制方程中的物理信息和适应边界条件，成为一种简便有效的PDE近似解法。含有狄拉克δ函数表示的点源项的PDEs在物理仿真中有很多应用，例如在电磁仿真中的脉冲激发电场或声学波动方程中的声源。然而，由于狄拉克δ函数带来的奇异性，传统的PINNs方法无法直接解决这类问题。现有方法（如Deep Ritz方法和NVIDIA SimNet的变分方法）需要将点源项转换为可计算形式，但并不适用于所有点源PDEs。

## 1.2 论文方法

​论文提出了一种通用的方法来解决基于PINNs方法的点源PDEs问题，不依赖任何标签数据或变分形式。提出了三项改进措施以提高准确性、效率和适用性：

（1）使用连续的单峰概率密度函数来模拟狄拉克δ函数，消除点源处的奇异性；

（2）提出带下界约束的不确定性加权算法，以平衡点源区域和其余区域的损失项；

（3）构建了一个多尺度DNN和周期激活函数的神经网络架构，以提高PINNs方法的准确性和收敛速度。

# 2  Mindspore实现PINNs求解二维带点源泊松方程



本案例采用MindFlow流体仿真套件，基于物理驱动的PINNs (Physics Informed Neural Networks)方法，求解二维带点源的泊松方程：
$$
\Delta u = - \delta(x-x_{src})\delta(y-y_{src}),
$$
其中$(x_{src}, y_{src})$为点源位置对应的坐标。

点源在数学上可以用狄拉克δ 函数来表示：
$$
\delta(x) = \begin{cases}
+\infty, & x = 0    \\
0,       & x \neq 0
\end{cases}
\qquad
\int_{-\infty}^{+\infty}\delta(x)dx = 1.
$$

## 2.1 准备工作

本案例要求 **MindSpore >= 2.0.0** 版本以调用如下接口: *mindspore.jit, mindspore.jit_class, mindspore.data_sink*。

此外，需要安装 **MindFlow >=0.1.0** 版本。

```
mindflow_version = "0.1.0"  # update if needed
# GPU Comment out the following code if you are using NPU.
!pip uninstall -y mindflow-gpu
!pip install mindflow-gpu==$mindflow_version

# NPU Uncomment if needed.
# !pip uninstall -y mindflow-ascend
# !pip install mindflow-ascend==$mindflow_version
```

本案例在求解域、边值条件、点源区域（以点源位置为中心的矩形区域）进行随机采样，生成训练数据集。

## 2.2 模型构建

本案例采用结合了sin激活函数的多尺度神经网络。

```
from mindflow.cell import MultiScaleFCSequential

# Create the model
model = MultiScaleFCSequential(config['model']['in_channels'],
                               config['model']['out_channels'],
                               config['model']['layers'],
                               config['model']['neurons'],
                               residual=True,
                               act=config['model']['activation'],
                               num_scales=config['model']['num_scales'],
                               amp_factor=1.0,
                               scale_factor=2.0,
                               input_scale=[10., 10.],
                               )
```



## 2.3 约束

在利用`mindflow`求解PDE时，我们需要写一个`mindflow.PDEWithLloss`的子类来定义控制方程和边界条件分别对应的损失函数项（`loss_pde`和`loss_bc`）。因为点源区域需要加密采样点，所以我们额外增加了一个损失函数项（`loss_src`）。

当PINNs方法将控制方程的残差作为损失函数项来约束神经网络时，狄拉克δ函数的奇异性使得神经网络的训练无法收敛，因此我们采用二维拉普拉斯分布的概率密度函数去近似狄拉克 δ 函数，即：
$$
\eta_{\alpha}(x, y) = \frac{1}{4\alpha^2} exp({-\frac{|x-x_{src}|+|y-y_{src}|}{\alpha}}) \qquad \underrightarrow{approx} \qquad \delta(x-x_{src})\delta(y-y_{src})
$$

其中 $\alpha$ 为核宽度。理论上来说，只要核宽度 $\alpha$充分小，那么上述概率密度函数就能很好地近似狄拉克δ 函数。但是实际上核宽度 $\alpha$的选取对于近似效果有着重要影响。当 $\alpha$太大时，概率密度函数 $\eta_{\alpha}(x, y)$与狄拉克 δ函数之间的近似误差会变大。但如果 $\alpha$ 太小，训练过程可能不会收敛，或者收敛后的精度可能很差。因此，  $\alpha$需要进行手工调参。我们这里将其确定为   $\alpha$=0.01。

```
import sympy
from mindspore import numpy as ms_np
from mindflow import PDEWithLoss, MTLWeightedLoss, sympy_to_mindspore

class Poisson(PDEWithLoss):
    """Define the loss of the Poisson equation."""

    def __init__(self, model):
        self.x, self.y = sympy.symbols("x y")
        self.u = sympy.Function("u")(self.x, self.y)
        self.in_vars = [self.x, self.y]
        self.out_vars = [self.u,]
        self.alpha = 0.01  # kernel width
        super(Poisson, self).__init__(model, self.in_vars, self.out_vars)
        self.bc_nodes = sympy_to_mindspore(self.bc(), self.in_vars, self.out_vars)
        self.loss_fn = MTLWeightedLoss(num_losses=3)

    def pde(self):
        """Define the gonvering equation."""
        uu_xx = sympy.diff(self.u, (self.x, 2))
        uu_yy = sympy.diff(self.u, (self.y, 2))

        # Use Laplace probability density function to approximate the Dirac \delta function.
        x_src = sympy.pi / 2
        y_src = sympy.pi / 2
        force_term = 0.25 / self.alpha**2 * sympy.exp(-(
            sympy.Abs(self.x - x_src) + sympy.Abs(self.y - y_src)) / self.alpha)

        poisson = uu_xx + uu_yy + force_term
        equations = {"poisson": poisson}
        return equations

    def bc(self):
        """Define the boundary condition."""
        bc_eq = self.u

        equations = {"bc": bc_eq}
        return equations

    def get_loss(self, pde_data, bc_data, src_data):
        """Define the loss function."""
        res_pde = self.parse_node(self.pde_nodes, inputs=pde_data)
        res_bc = self.parse_node(self.bc_nodes, inputs=bc_data)
        res_src = self.parse_node(self.pde_nodes, inputs=src_data)

        loss_pde = ms_np.mean(ms_np.square(res_pde[0]))
        loss_bc = ms_np.mean(ms_np.square(res_bc[0]))
        loss_src = ms_np.mean(ms_np.square(res_src[0]))

        return self.loss_fn((loss_pde, loss_bc, loss_src))

# Create the problem and optimizer
problem = Poisson(model)
```

```
poisson: Derivative(u(x, y), (x, 2)) + Derivative(u(x, y), (y, 2)) + 2500.0*exp(-100.0*Abs(x - pi/2))*exp(-100.0*Abs(y - pi/2))
    Item numbers of current derivative formula nodes: 3
bc: u(x, y)
    Item numbers of current derivative formula nodes: 1
```



## 2.4 优化器

本案例采用Adam优化器，并在训练进行到40%、60%、80%时，学习率衰减为初始学习率的1/10、1/100、1/1000。



```
n_epochs = 250

params = model.trainable_params() + problem.loss_fn.trainable_params()
steps_per_epoch = ds_train.get_dataset_size()
milestone = [int(steps_per_epoch * n_epochs * x) for x in [0.4, 0.6, 0.8]]
lr_init = config["optimizer"]["initial_lr"]
learning_rates = [lr_init * (0.1**x) for x in [0, 1, 2]]
lr_ = nn.piecewise_constant_lr(milestone, learning_rates)
optimizer = nn.Adam(params, learning_rate=lr_)
```



## 2.5 模型训练

使用MindSpore>= 2.0.0的版本，可以使用函数式编程范式训练神经网络。

```
def train():
    grad_fn = ops.value_and_grad(problem.get_loss, None, optimizer.parameters, has_aux=False)
    
    use_ascend = False

    @jit
    def train_step(pde_data, bc_data, src_data):
        loss, grads = grad_fn(pde_data, bc_data, src_data)
        # if use_ascend:
        #     loss = loss_scaler.unscale(loss)
        #     is_finite = all_finite(grads)
        #     if is_finite:
        #         grads = loss_scaler.unscale(grads)
        #         loss = ops.depend(loss, optimizer(grads))
        #     loss_scaler.adjust(is_finite)
        # else:
        #     loss = ops.depend(loss, optimizer(grads))
        loss = ops.depend(loss, optimizer(grads))
        return loss

    def train_epoch(model, dataset, i_epoch):
        local_time_beg = time.time()

        model.set_train()
        for _, (pde_data, bc_data, src_data) in enumerate(dataset):
            loss = train_step(pde_data, bc_data, src_data)

        print(
            f"epoch: {i_epoch} train loss: {float(loss):.8f}" +
            f" epoch time: {time.time() - local_time_beg:.2f}s")

    for i_epoch in range(1, 1 + n_epochs):
        train_epoch(model, ds_train, i_epoch)

time_beg = time.time()
train()
print(f"End-to-End total time: {time.time() - time_beg:.1f} s")
```

```
[ERROR] CORE(7707,7fd299739740,python):2024-06-12-10:40:59.437.246 [mindspore/core/utils/file_utils.cc:253] GetRealPath] Get realpath failed, path[/tmp/ipykernel_7707/1406912981.py]
[ERROR] CORE(7707,7fd299739740,python):2024-06-12-10:40:59.437.335 [mindspore/core/utils/file_utils.cc:253] GetRealPath] Get realpath failed, path[/tmp/ipykernel_7707/1406912981.py]
[ERROR] CORE(7707,7fd299739740,python):2024-06-12-10:40:59.437.371 [mindspore/core/utils/file_utils.cc:253] GetRealPath] Get realpath failed, path[/tmp/ipykernel_7707/1406912981.py]
[ERROR] CORE(7707,7fd299739740,python):2024-06-12-10:40:59.459.259 [mindspore/core/utils/file_utils.cc:253] GetRealPath] Get realpath failed, path[/tmp/ipykernel_7707/1406912981.py]
[ERROR] CORE(7707,7fd299739740,python):2024-06-12-10:40:59.461.789 [mindspore/core/utils/file_utils.cc:253] GetRealPath] Get realpath failed, path[/tmp/ipykernel_7707/1406912981.py]
```

​        

```
epoch: 1 train loss: 22999.73437500 epoch time: 29.42s
epoch: 2 train loss: 19637.60156250 epoch time: 15.86s
epoch: 3 train loss: 13385.56640625 epoch time: 15.87s
epoch: 4 train loss: 13447.35351562 epoch time: 15.84s
epoch: 5 train loss: 7991.54443359 epoch time: 15.84s
epoch: 6 train loss: 5600.45117188 epoch time: 15.86s
epoch: 7 train loss: 3709.35766602 epoch time: 15.90s
epoch: 8 train loss: 2007.01782227 epoch time: 15.88s
epoch: 9 train loss: 1765.28381348 epoch time: 15.90s
epoch: 10 train loss: 1757.24768066 epoch time: 15.87s
epoch: 11 train loss: 1964.14221191 epoch time: 15.87s
epoch: 12 train loss: 1459.37060547 epoch time: 15.87s
epoch: 13 train loss: 1343.92114258 epoch time: 15.88s
epoch: 14 train loss: 1347.97863770 epoch time: 15.83s
epoch: 15 train loss: 898.37341309 epoch time: 15.87s
epoch: 16 train loss: 627.40295410 epoch time: 15.90s
epoch: 17 train loss: 1308.07080078 epoch time: 15.89s
epoch: 18 train loss: 786.27917480 epoch time: 15.94s
epoch: 19 train loss: 863.24548340 epoch time: 15.86s
epoch: 20 train loss: 884.04187012 epoch time: 15.84s
epoch: 21 train loss: 542.18591309 epoch time: 15.90s
epoch: 22 train loss: 396.93612671 epoch time: 15.88s
epoch: 23 train loss: 464.11218262 epoch time: 15.87s
epoch: 24 train loss: 356.23913574 epoch time: 15.85s
epoch: 25 train loss: 607.33758545 epoch time: 15.88s
epoch: 26 train loss: 284.36099243 epoch time: 15.86s
epoch: 27 train loss: 254.90711975 epoch time: 15.86s
epoch: 28 train loss: 308.29888916 epoch time: 15.90s
epoch: 29 train loss: 250.09461975 epoch time: 16.01s
epoch: 30 train loss: 201.58364868 epoch time: 15.90s
epoch: 31 train loss: 1278.80102539 epoch time: 15.90s
epoch: 32 train loss: 290.41802979 epoch time: 15.86s
epoch: 33 train loss: 347.99600220 epoch time: 15.88s
epoch: 34 train loss: 638.98297119 epoch time: 15.92s
epoch: 35 train loss: 328.75177002 epoch time: 15.92s
epoch: 36 train loss: 382.20758057 epoch time: 15.93s
epoch: 37 train loss: 532.12603760 epoch time: 15.90s
epoch: 38 train loss: 333.04595947 epoch time: 15.84s
epoch: 39 train loss: 155.10746765 epoch time: 15.87s
epoch: 40 train loss: 538.06994629 epoch time: 15.87s
epoch: 41 train loss: 129.63583374 epoch time: 15.85s
epoch: 42 train loss: 123.82033539 epoch time: 15.86s
epoch: 43 train loss: 195.79432678 epoch time: 15.88s
epoch: 44 train loss: 516.09313965 epoch time: 15.90s
epoch: 45 train loss: 657.38629150 epoch time: 15.88s
epoch: 46 train loss: 315.24624634 epoch time: 15.89s
epoch: 47 train loss: 388.37811279 epoch time: 15.88s
epoch: 48 train loss: 81.73287201 epoch time: 15.90s
epoch: 49 train loss: 114.90600586 epoch time: 15.86s
epoch: 50 train loss: 90.29770660 epoch time: 15.86s
epoch: 51 train loss: 88.09072113 epoch time: 15.91s
epoch: 52 train loss: 276.52243042 epoch time: 15.88s
epoch: 53 train loss: 120.67494202 epoch time: 15.92s
epoch: 54 train loss: 61.28784561 epoch time: 15.92s
epoch: 55 train loss: 135.42512512 epoch time: 15.90s
epoch: 56 train loss: 42.80917740 epoch time: 15.93s
epoch: 57 train loss: 121.44651031 epoch time: 15.93s
epoch: 58 train loss: 36.30596161 epoch time: 15.86s
epoch: 59 train loss: 47.89866638 epoch time: 15.88s
epoch: 60 train loss: 61.14999390 epoch time: 15.87s
epoch: 61 train loss: 178.60267639 epoch time: 15.84s
epoch: 62 train loss: 133.30516052 epoch time: 15.86s
epoch: 63 train loss: 56.35380554 epoch time: 15.87s
epoch: 64 train loss: 100.39230347 epoch time: 15.88s
epoch: 65 train loss: 46.90230179 epoch time: 15.85s
epoch: 66 train loss: 122.71855164 epoch time: 15.86s
epoch: 67 train loss: 74.45882416 epoch time: 15.90s
epoch: 68 train loss: 25.49005127 epoch time: 15.85s
epoch: 69 train loss: 57.50381851 epoch time: 15.86s
epoch: 70 train loss: 107.04942322 epoch time: 15.86s
epoch: 71 train loss: 41.09004211 epoch time: 15.85s
epoch: 72 train loss: 110.52747345 epoch time: 15.87s
epoch: 73 train loss: 59.60391617 epoch time: 15.91s
epoch: 74 train loss: 56.73511124 epoch time: 15.93s
epoch: 75 train loss: 44.33506775 epoch time: 15.90s
epoch: 76 train loss: 101.40853882 epoch time: 15.87s
epoch: 77 train loss: 67.73876190 epoch time: 15.88s
epoch: 78 train loss: 57.76534653 epoch time: 15.91s
epoch: 79 train loss: 22.53282928 epoch time: 15.93s
epoch: 80 train loss: 60.14231491 epoch time: 15.90s
epoch: 81 train loss: 33.90668488 epoch time: 15.92s
epoch: 82 train loss: 26.14128304 epoch time: 15.88s
epoch: 83 train loss: 44.27367783 epoch time: 15.87s
epoch: 84 train loss: 58.28453445 epoch time: 15.86s
epoch: 85 train loss: 23.80356598 epoch time: 15.87s
epoch: 86 train loss: 35.77706909 epoch time: 15.87s
epoch: 87 train loss: 29.04700851 epoch time: 15.85s
epoch: 88 train loss: 71.09509277 epoch time: 15.89s
epoch: 89 train loss: 35.32649612 epoch time: 15.86s
epoch: 90 train loss: 49.14509201 epoch time: 15.89s
epoch: 91 train loss: 41.91509247 epoch time: 15.83s
epoch: 92 train loss: 80.38156891 epoch time: 15.93s
epoch: 93 train loss: 62.56001663 epoch time: 15.89s
epoch: 94 train loss: 28.78996086 epoch time: 15.88s
epoch: 95 train loss: 61.43371201 epoch time: 15.86s
epoch: 96 train loss: 32.91157150 epoch time: 15.84s
epoch: 97 train loss: 35.38578796 epoch time: 15.90s
epoch: 98 train loss: 40.13176346 epoch time: 15.88s
epoch: 99 train loss: 20.90218353 epoch time: 15.90s
epoch: 100 train loss: 10.11530495 epoch time: 15.86s
epoch: 101 train loss: 9.24770164 epoch time: 15.88s
epoch: 102 train loss: 9.06866264 epoch time: 15.86s
epoch: 103 train loss: 8.73276138 epoch time: 15.91s
epoch: 104 train loss: 8.41265297 epoch time: 15.89s
epoch: 105 train loss: 8.54669380 epoch time: 15.86s
epoch: 106 train loss: 9.01762867 epoch time: 15.87s
epoch: 107 train loss: 8.58020782 epoch time: 15.89s
epoch: 108 train loss: 8.64859581 epoch time: 15.87s
epoch: 109 train loss: 8.90017796 epoch time: 15.92s
epoch: 110 train loss: 9.63255596 epoch time: 15.90s
epoch: 111 train loss: 8.64641953 epoch time: 15.93s
epoch: 112 train loss: 8.81948948 epoch time: 15.99s
epoch: 113 train loss: 8.74738503 epoch time: 16.00s
epoch: 114 train loss: 8.54762268 epoch time: 15.91s
epoch: 115 train loss: 8.37177372 epoch time: 15.95s
epoch: 116 train loss: 8.05216217 epoch time: 15.92s
epoch: 117 train loss: 8.14800739 epoch time: 15.86s
epoch: 118 train loss: 7.92260981 epoch time: 15.86s
epoch: 119 train loss: 7.95822811 epoch time: 15.90s
epoch: 120 train loss: 7.84642029 epoch time: 15.87s
epoch: 121 train loss: 9.27675819 epoch time: 15.85s
epoch: 122 train loss: 7.81504250 epoch time: 15.89s
epoch: 123 train loss: 7.31824350 epoch time: 15.85s
epoch: 124 train loss: 7.69555330 epoch time: 15.85s
epoch: 125 train loss: 8.32190418 epoch time: 15.89s
epoch: 126 train loss: 8.32282257 epoch time: 15.85s
epoch: 127 train loss: 7.55847359 epoch time: 15.86s
epoch: 128 train loss: 7.27688313 epoch time: 15.85s
epoch: 129 train loss: 7.46255398 epoch time: 15.86s
epoch: 130 train loss: 8.60893059 epoch time: 15.91s
epoch: 131 train loss: 7.38764095 epoch time: 15.97s
epoch: 132 train loss: 6.92643452 epoch time: 15.90s
epoch: 133 train loss: 8.29349327 epoch time: 15.85s
epoch: 134 train loss: 6.79966879 epoch time: 15.88s
epoch: 135 train loss: 7.07550335 epoch time: 15.90s
epoch: 136 train loss: 7.78727913 epoch time: 15.90s
epoch: 137 train loss: 8.05565929 epoch time: 15.89s
epoch: 138 train loss: 6.99063587 epoch time: 15.86s
epoch: 139 train loss: 7.29381514 epoch time: 15.89s
epoch: 140 train loss: 7.77537251 epoch time: 15.89s
epoch: 141 train loss: 7.15032482 epoch time: 15.85s
epoch: 142 train loss: 6.74619627 epoch time: 15.86s
epoch: 143 train loss: 6.90080452 epoch time: 15.83s
epoch: 144 train loss: 9.95440388 epoch time: 15.83s
epoch: 145 train loss: 6.59037066 epoch time: 15.84s
epoch: 146 train loss: 6.82514524 epoch time: 15.80s
epoch: 147 train loss: 6.51371956 epoch time: 15.86s
epoch: 148 train loss: 6.25416040 epoch time: 15.91s
epoch: 149 train loss: 6.84505510 epoch time: 15.80s
epoch: 150 train loss: 7.75001383 epoch time: 15.81s
epoch: 151 train loss: 6.29045677 epoch time: 15.83s
epoch: 152 train loss: 6.40268517 epoch time: 15.75s
epoch: 153 train loss: 6.16711521 epoch time: 15.76s
epoch: 154 train loss: 6.48522472 epoch time: 15.77s
epoch: 155 train loss: 6.23228121 epoch time: 15.72s
epoch: 156 train loss: 6.17456627 epoch time: 15.71s
epoch: 157 train loss: 6.14163113 epoch time: 15.71s
epoch: 158 train loss: 6.56846237 epoch time: 15.72s
epoch: 159 train loss: 6.33627796 epoch time: 15.71s
epoch: 160 train loss: 6.46562195 epoch time: 15.71s
epoch: 161 train loss: 6.36947489 epoch time: 15.74s
epoch: 162 train loss: 6.29502916 epoch time: 15.71s
epoch: 163 train loss: 6.37959146 epoch time: 15.73s
epoch: 164 train loss: 6.37263155 epoch time: 15.69s
epoch: 165 train loss: 6.42197657 epoch time: 15.76s
epoch: 166 train loss: 6.24008703 epoch time: 15.72s
epoch: 167 train loss: 6.29987764 epoch time: 15.69s
epoch: 168 train loss: 6.35437012 epoch time: 15.72s
epoch: 169 train loss: 6.69032860 epoch time: 15.73s
epoch: 170 train loss: 6.53317928 epoch time: 15.73s
epoch: 171 train loss: 6.68145990 epoch time: 15.73s
epoch: 172 train loss: 6.34727001 epoch time: 15.68s
epoch: 173 train loss: 6.84885454 epoch time: 15.72s
epoch: 174 train loss: 6.55553293 epoch time: 15.71s
epoch: 175 train loss: 6.17183590 epoch time: 15.69s
epoch: 176 train loss: 6.59528351 epoch time: 15.74s
epoch: 177 train loss: 6.17375755 epoch time: 15.71s
epoch: 178 train loss: 6.27593565 epoch time: 15.69s
epoch: 179 train loss: 6.07295465 epoch time: 15.70s
epoch: 180 train loss: 6.42197466 epoch time: 15.74s
epoch: 181 train loss: 6.74848270 epoch time: 15.71s
epoch: 182 train loss: 6.22701263 epoch time: 15.72s
epoch: 183 train loss: 6.28402424 epoch time: 15.69s
epoch: 184 train loss: 6.15117359 epoch time: 15.72s
epoch: 185 train loss: 6.23298502 epoch time: 15.77s
epoch: 186 train loss: 6.27542830 epoch time: 15.72s
epoch: 187 train loss: 6.16910934 epoch time: 15.75s
epoch: 188 train loss: 6.12605381 epoch time: 15.77s
epoch: 189 train loss: 6.38213634 epoch time: 15.74s
epoch: 190 train loss: 6.22025108 epoch time: 15.74s
epoch: 191 train loss: 6.18379784 epoch time: 15.69s
epoch: 192 train loss: 6.34441423 epoch time: 15.77s
epoch: 193 train loss: 6.08603907 epoch time: 15.73s
epoch: 194 train loss: 6.46428061 epoch time: 15.75s
epoch: 195 train loss: 6.46370602 epoch time: 15.74s
epoch: 196 train loss: 6.46293545 epoch time: 15.73s
epoch: 197 train loss: 6.23089695 epoch time: 15.72s
epoch: 198 train loss: 6.29760361 epoch time: 15.72s
epoch: 199 train loss: 5.96060467 epoch time: 15.71s
epoch: 200 train loss: 6.30901146 epoch time: 15.73s
epoch: 201 train loss: 5.98970032 epoch time: 15.71s
epoch: 202 train loss: 6.01230955 epoch time: 15.71s
epoch: 203 train loss: 6.20635366 epoch time: 15.70s
epoch: 204 train loss: 5.95029068 epoch time: 15.70s
epoch: 205 train loss: 6.07702541 epoch time: 15.70s
epoch: 206 train loss: 6.42635345 epoch time: 15.78s
epoch: 207 train loss: 6.18799448 epoch time: 15.76s
epoch: 208 train loss: 6.44426441 epoch time: 15.74s
epoch: 209 train loss: 6.37166405 epoch time: 15.79s
epoch: 210 train loss: 6.05553675 epoch time: 15.71s
epoch: 211 train loss: 6.08306313 epoch time: 15.72s
epoch: 212 train loss: 6.24026632 epoch time: 15.74s
epoch: 213 train loss: 6.39114189 epoch time: 15.76s
epoch: 214 train loss: 6.19074726 epoch time: 15.69s
epoch: 215 train loss: 6.43641329 epoch time: 15.78s
epoch: 216 train loss: 6.22963047 epoch time: 15.77s
epoch: 217 train loss: 6.06437159 epoch time: 15.75s
epoch: 218 train loss: 6.06974363 epoch time: 15.71s
epoch: 219 train loss: 6.08114624 epoch time: 15.72s
epoch: 220 train loss: 5.96954155 epoch time: 15.73s
epoch: 221 train loss: 6.40169811 epoch time: 15.71s
epoch: 222 train loss: 6.53714705 epoch time: 15.71s
epoch: 223 train loss: 6.25920582 epoch time: 15.74s
epoch: 224 train loss: 6.03552437 epoch time: 15.76s
epoch: 225 train loss: 6.32021332 epoch time: 15.73s
epoch: 226 train loss: 6.10254860 epoch time: 15.76s
epoch: 227 train loss: 6.08055782 epoch time: 15.79s
epoch: 228 train loss: 6.08783770 epoch time: 15.76s
epoch: 229 train loss: 6.14675379 epoch time: 15.71s
epoch: 230 train loss: 6.30144310 epoch time: 15.80s
epoch: 231 train loss: 6.10744953 epoch time: 15.71s
epoch: 232 train loss: 6.01194239 epoch time: 15.72s
epoch: 233 train loss: 6.21889305 epoch time: 15.77s
epoch: 234 train loss: 6.54785824 epoch time: 15.74s
epoch: 235 train loss: 6.31049299 epoch time: 15.75s
epoch: 236 train loss: 6.10820484 epoch time: 15.77s
epoch: 237 train loss: 6.22016859 epoch time: 15.71s
epoch: 238 train loss: 6.60480738 epoch time: 15.72s
epoch: 239 train loss: 6.15671301 epoch time: 15.73s
epoch: 240 train loss: 6.16796589 epoch time: 15.70s
epoch: 241 train loss: 5.95938063 epoch time: 15.70s
epoch: 242 train loss: 6.19034529 epoch time: 15.75s
epoch: 243 train loss: 6.14158058 epoch time: 15.71s
epoch: 244 train loss: 6.41351843 epoch time: 15.78s
epoch: 245 train loss: 6.18988800 epoch time: 15.79s
epoch: 246 train loss: 6.24767113 epoch time: 15.78s
epoch: 247 train loss: 6.05907440 epoch time: 15.75s
epoch: 248 train loss: 6.14543343 epoch time: 15.71s
epoch: 249 train loss: 6.39600611 epoch time: 15.71s
epoch: 250 train loss: 5.99916840 epoch time: 15.72s
End-to-End total time: 3969.0 s
```



## 2.6 模型推理及可视化

计算相对L2误差以及绘制参考解和模型预测结果的对比图。

```
from src.utils import calculate_l2_error, visual

# Create the dataset
ds_test = create_test_dataset(config)

# Evaluate the model
calculate_l2_error(model, ds_test)

# Visual comparison of label and prediction
visual(model, ds_test)
```

```
Relative L2 error:   0.0106
```

