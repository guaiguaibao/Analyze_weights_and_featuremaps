from alexnet_model import AlexNet_v1, AlexNet_v2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import Model, Input

im_height = 224
im_width = 224

# load image
img = Image.open("../tulip.jpg")
# resize image to 224x224
img = img.resize((im_width, im_height))

# scaling pixel value to (0-1)
img = np.array(img) / 255.

# Add the image to a batch where it's the only member.
img = (np.expand_dims(img, 0))


model = AlexNet_v1(class_num=5)  # functional api
# # 使用子类方式构建模型
# model = AlexNet_v2(class_num=5)  # subclass api
# model.build((None, 224, 224, 3))

# If `by_name` is False weights are loaded based on the network's topology.
# 在修改了我们关注的层的名称之后，并没有影响模型权重的加载，是因为load_weights函数有一个参数是by_name，这个参数默认是False，也就是基于网络的拓扑结构进行加载权重，所以修改层名称是没有关系的
model.load_weights("./myAlex.h5")
# model.load_weights("./submodel.h5")
# for layer in model.layers:
#     print(layer.name)
model.summary()
# layers_name是要查看feature map的卷积层，这里的卷积层的名称是默认生成的，不是人为设定的，如果想要自己设定，那么需要在model.py脚本中创建卷积层的地方传入name参数，而且需要将所有要查看的层名称都要人为设定名称，因为如果只设定一个，其他的层名称默认名称随之发生变化。
layers_name = ["conv2d", "conv2d_1"]

# 使用functional和subclassing得到特征图的方式不一样
# functional API
try:
    # input_node是模型的输入层
    input_node = model.input
    # 通过get_layer获取对应的网络层，然后通过output属性获得该层的输出
    output_node = [model.get_layer(name=layer_name).output for layer_name in layers_name]
    # 重新定义一个模型，单输入多输出
    model1 = Model(inputs=input_node, outputs=output_node)
    # 这里的多个输出就是目标网络层的输出特征图
    outputs = model1.predict(img)
    for index, feature_map in enumerate(outputs):
        # [N, H, W, C] -> [H, W, C]
        im = np.squeeze(feature_map)

        # show top 12 feature maps，注意要小于输出特征图的通道个数
        plt.figure()
        for i in range(12):
            ax = plt.subplot(3, 4, i + 1)
            # [H, W, C]
            # 设置cmap为gray，也就是以从白色到黑色过渡的灰度图像展示，如果不设置该参数，展示的图像是从蓝色到绿色过渡的图像，虽然也是灰度图像，
            plt.imshow(im[:, :, i], cmap='gray')
        plt.suptitle(layers_name[index])
        plt.show()
except Exception as e:
    print(e)

# subclasses API
# outputs = model.receive_feature_map(img, layers_name)
# for index, feature_maps in enumerate(outputs):
#     # [N, H, W, C] -> [H, W, C]
#     im = np.squeeze(feature_maps)
#
#     # show top 12 feature maps
#     plt.figure()
#     for i in range(12):
#         ax = plt.subplot(3, 4, i + 1)
#         # [H, W, C]
#         plt.imshow(im[:, :, i], cmap='gray')
#     plt.suptitle(layers_name[index])
#     plt.show()
