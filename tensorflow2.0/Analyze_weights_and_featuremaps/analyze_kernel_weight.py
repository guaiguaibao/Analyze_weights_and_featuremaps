from alexnet_model import AlexNet_v1, AlexNet_v2
import numpy as np
import matplotlib.pyplot as plt

model = AlexNet_v1(class_num=5)  # functional api
# model = AlexNet_v2(class_num=5)  # subclass api
# model.build((None, 224, 224, 3))
model.load_weights("./myAlex.h5")
# model.load_weights("./submodel.h5")
model.summary()

# 使用functional和subclassing得到权重的方式相同
for layer in model.layers:
    for index, weight in enumerate(layer.weights):
        # tensorflow中某一层的权重矩阵的各个维度如下：
        # [kernel_height, kernel_width, kernel_channel, kernel_number]

        # 注意这里没有将每个卷积核分开，而是将每一层的所有卷积核的参数放在一起进行统计和展示的，实际上应该将每一层的每一个卷积核进行分别统计和展示
        weight_t = weight.numpy()
        # read a kernel information
        # 得到某一个卷积核的参数直接在最后一个维度取出对应的卷积核就行
        # k = weight_t[:, :, :, 0]

        # calculate mean, std, min, max
        weight_mean = weight_t.mean()
        weight_std = weight_t.std(ddof=1)
        weight_min = weight_t.min()
        weight_max = weight_t.max()
        print("mean is {}, std is {}, min is {}, max is {}".format(weight_mean,
                                                                   weight_std,
                                                                   weight_max,
                                                                   weight_min))

        # plot hist image
        plt.close()
        # 通过reshape将一层的权重矩阵展平成一个一维向量
        weight_vec = np.reshape(weight_t, [-1])
        # 设置bins，是将min到max之间平均分成50份，然后统计落在每一份中的参数个数
        plt.hist(weight_vec, bins=50)
        plt.title(weight.name)
        plt.show()
