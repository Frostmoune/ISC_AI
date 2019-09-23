import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import variable_scope
from tensorflow.contrib.layers.python.layers import utils
from .common_helpers import ensure_type
import horovod.tensorflow as hvd
import math

# activation，只支持relu和lrelu两种
def activate(type_='relu'):
    if type_ == 'relu':
        act_fn = tf.nn.relu
    elif type_ == 'lrelu':
        act_fn = tf.nn.leaky_relu
    else:
        raise TypeError("Unsupported activation type")
    return act_fn

# upsample，支持bilinear、nearest和deconv三种
# in_: (N, H, W, C)
# size_为要扩大到的size，如size_=(X, Y)，则out: (N, X, Y, C)
# nc和stride参数只供deconv使用，nc代表通道数
def UpSample(in_, size_, type_='bilinear', name="upsample", nc=256, stride=2):
    if type_ == 'bilinear':
        return tf.image.resize_bilinear(in_, size_, name=name)
    elif type_ == 'nearest':
        return tf.image.resize_nearest_neighbor(in_, size_, name=name)
    elif type_ == 'deconv':
        return slim.conv2d_transpose(in_, nc, [3, 3], padding='SAME', stride=stride, scope=name,
                                    weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                    weights_regularizer=slim.l2_regularizer(0.0005))
    else:
        raise TypeError("Unsupported upsample type")

# n个conv+UpSample2x(扩大两倍)拼接
def UpSampleBlock(net, out_size, n, act='relu', type_='bilinear', name="upsample", nc=256, stride=2):
    act_fn = activate(type_=act)
    with slim.arg_scope([slim.conv2d], 
                        activation_fn=act_fn,
                        weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                        weights_regularizer=slim.l2_regularizer(0.0005),
                        padding='SAME'):
        for i in range(n):
            now_size = out_size//(2**(n-1-i))
            net = slim.conv2d(net, nc, 3, stride=1)
            net = UpSample(net, now_size, type_=type_, name="%s_%d"%(name, i))
        return net

# x: (N, H, W, C)
# in_nc: dense_block最后一个conv和shortcut的conv的通道数
# out_nc: 最后一个conv的通道数，默认与in_nc相等
# nconv: dense_block中conv的数量
# gc: dense_block中间conv的通道数
# kernel_size: 除最后一个conv之外的conv的大小
# stride：倒数第二个conv和和shortcut的conv的stride
# act: 激活函数种类
# type_: 分为concat和add两种
def RDB(x, in_nc, out_nc=None, nconv=5, gc=32, kernel_size=[3, 3], 
        stride=1, act='relu', name='RDB', type_='concat'):
    act_fn = activate(type_=act)
    
    if out_nc is None:
        out_nc = in_nc
    in_x = x
    with slim.arg_scope([slim.conv2d], 
                        activation_fn=act_fn,
                        weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                        weights_regularizer=slim.l2_regularizer(0.0005),
                        padding='SAME'):
        # dense_block部分
        for i in range(nconv):
            if i < nconv-1:
                tmp = slim.conv2d(x, gc, kernel_size, scope='%s/conv%d'%(name, i))
                x = tf.concat([x, tmp], axis=3)
            else:
                x = slim.conv2d(x, in_nc, kernel_size, scope='%s/conv%d'%(name, i))

        # 倒数第二个conv和shortcut的conv可调整stride
        x = slim.conv2d(x, in_nc, kernel_size, stride=stride, scope='%s/conv%d'%(name, nconv))
        shortcut = slim.conv2d(in_x, in_nc, kernel_size, stride=stride, scope='%s/conv_shortcut'%(name))
        if type_ == 'concat':
            x = tf.concat([x, shortcut], axis=3)
        else:
            x = tf.add(x, shortcut)
        # 最后一个带BN的conv1x1
        x = slim.conv2d(x, out_nc, [1, 1], stride=1, normalizer_fn=slim.batch_norm, scope='%s/conv%d'%(name, nconv+1))
        return x

# net: (N, H, W, C)
# in_nc: RDB的in_nc
# nb: RDB的数量
# stride: 第一个RDB的stride
# 其余参数见RDB的参数介绍
def RRDB(in_, in_nc, nb, nconv=5, gc=32, kernel_size=[3, 3], 
        stride=1, act='relu', name='RRDB'):

    net = in_
    # 第一个RDB, 可调整stride
    net = RDB(net, in_nc, out_nc=in_nc*2, nconv=nconv, gc=gc, 
            kernel_size=kernel_size, stride=stride, act=act, name='%s/RDB%d'%(name, nb))
    # stride不同时shortcut连接的位置也不同
    if stride == 1:
        tmp = in_
    else:
        tmp = net

    for i in range(nb-1):
        net = RDB(net, in_nc, nconv=nconv, gc=gc, kernel_size=kernel_size, 
                stride=1, act=act, name='%s/RDB%d'%(name, i+1))

    # shortcut
    if nb > 1:
        shortcut = slim.conv2d(tmp, in_nc, kernel_size, stride=1, scope='%s/conv_shortcut'%(name))
        net = tf.concat([net, shortcut], axis=3)
        # concat之后带BN的conv1x1
        net = slim.conv2d(net, in_nc, [1, 1], normalizer_fn=slim.batch_norm, stride=1, scope='%s/conv1x1'%(name))

    return net

# 跟SKConv有关的block请参考https://arxiv.org/abs/1903.06586这篇论文
# 和https://github.com/pppLang/SKNet这个github
def SKConv(in_, nc, nconv=2, r=2, stride=1, L=32, act='relu', name="SKConv", normalize=True):
    act_fn = activate(type_=act)
    W = in_.shape[1]
    H = in_.shape[2]

    nc = int(nc)
    d = max(nc//r, L)
    with slim.arg_scope([slim.conv2d, slim.fully_connected], 
                        activation_fn=act_fn,
                        weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                        weights_regularizer=slim.l2_regularizer(0.0005)):
        with slim.arg_scope([slim.conv2d], padding='SAME'):
            feas = None
            for i in range(nconv):
                fea = slim.conv2d(in_, nc, 2*i+3, normalizer_fn=slim.batch_norm, stride=stride, scope='%s/conv%d'%(name, i))
                fea = tf.expand_dims(fea, axis=-1)
                if i == 0:
                    feas = fea
                else:
                    feas = tf.concat([feas, fea], axis=-1)
            
            fea_U = tf.reduce_sum(feas, axis=-1)
            fea_s = slim.avg_pool2d(fea_U, [W//stride, H//stride], scope='%s/avgPool'%(name))
            fea_s = tf.squeeze(fea_s)
            fea_z = slim.fully_connected(fea_s, d)
            
            attention_vectors = None
            for i in range(nconv):
                vector = slim.fully_connected(fea_z, nc)
                vector = tf.expand_dims(vector, axis=-1)
                if i == 0:
                    attention_vectors = vector
                else:
                    attention_vectors = tf.concat([attention_vectors, vector], axis=-1)

            attention_vectors = tf.expand_dims(tf.nn.softmax(attention_vectors, axis=-1), axis=1)
            attention_vectors = tf.expand_dims(attention_vectors, axis=1)
            fea_v = feas * attention_vectors
            fea_v = tf.reduce_sum(fea_v, axis=-1)
            if normalize:
                fea_v = slim.batch_norm(fea_v)
            return fea_v

# 跟SKConv有关的block请参考https://arxiv.org/abs/1903.06586这篇论文
# 和https://github.com/pppLang/SKNet这个github
def SKUnit(in_, out_nc, mid_nc=None, nconv=2, r=2, stride=1, L=32, act='relu', name="SKUnit"):
    act_fn = activate(type_=act)
    if mid_nc is None:
        mid_nc = in_.shape[3] // 2
    with slim.arg_scope([slim.conv2d], 
                        activation_fn=act_fn,
                        weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                        weights_regularizer=slim.l2_regularizer(0.0005),
                        normalizer_fn=slim.batch_norm,
                        padding='SAME'):
        net = slim.conv2d(in_, mid_nc, 1, stride=1, scope="%s/conv1"%(name))
        net = SKConv(net, mid_nc, nconv=nconv, r=r, L=L, stride=stride, act=act, name="%s/SKConv"%(name))
        net = slim.conv2d(net, out_nc, 1, stride=1, scope="%s/conv2"%(name))

        tmp = slim.conv2d(in_, out_nc, 1, stride=stride, scope="%s/conv_shortcut"%(name))
        net = tf.concat([net, tmp], axis=3)
        net = slim.conv2d(net, out_nc, 1, stride=1, scope="%s/conv3"%(name))
        return net

def SKBlock(net, out_nc, nb, nconv=2, stride=1, r=2, L=32, act='relu', name='SKBlock'):
    tmp = net
    net = SKUnit(net, out_nc, nconv=nconv, stride=stride, act=act, name='%s/SKUnit1'%(name))
    for i in range(nb-1):
        net = SKUnit(net, out_nc, nconv=nconv, stride=1, act=act, name='%s/SKUnit%d'%(name, i+2))

    if nb > 1:
        shortcut = slim.conv2d(tmp, out_nc, 1, stride=stride, scope='%s/conv_shortcut'%(name))
        net = tf.concat([net, shortcut], axis=3)
        net = slim.conv2d(net, out_nc, 1, normalizer_fn=slim.batch_norm, stride=1, scope='%s/conv1x1'%(name))
    
    return net

# SKDense就是用SKConv加Dense操作，并没有什么效果
def SKDenseUnit(in_, out_nc, nd=3, nconv=2, gc=32, r=2, stride=1, L=32, act='relu', name="SKDenseUnit"):
    act_fn = activate(type_=act)
    with slim.arg_scope([slim.conv2d], 
                        activation_fn=act_fn,
                        weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                        weights_regularizer=slim.l2_regularizer(0.0005),
                        padding='SAME'):
        net = in_
        tmp = slim.conv2d(net, gc, 1, stride=1, scope="%s/conv1"%(name))
        net = tf.concat([net, tmp], axis=3)
        for i in range(nd-2):
            tmp = SKConv(net, gc, nconv=nconv, r=r, L=L, stride=1, act=act, normalize=False, 
                            name="%s/SKConv%d"%(name, i))
            net = tf.concat([net, tmp], axis=3)
        net = slim.conv2d(net, out_nc, 1, stride=stride, scope="%s/conv2"%(name))

        tmp = slim.conv2d(in_, out_nc, 1, stride=stride, scope="%s/conv_shortcut"%(name))
        net = tf.concat([net, tmp], axis=3)
        net = slim.conv2d(net, out_nc, 1, stride=1, normalizer_fn=slim.batch_norm, scope="%s/conv3"%(name))
        return net

def SKDense(net, out_nc, nb, nd=3, nconv=2, gc=32, stride=1, r=2, L=32, act='relu', name='SKDense'):
    tmp = net
    net = SKDenseUnit(net, out_nc, nd=nd, nconv=nconv, gc=gc, stride=stride, act=act, name='%s/SKUnit1'%(name))
    for i in range(nb-1):
        net = SKDenseUnit(net, out_nc, nd=nd, nconv=nconv, gc=gc, stride=1, act=act, name='%s/SKUnit%d'%(name, i+2))

    if nb > 1:
        shortcut = slim.conv2d(tmp, out_nc, 1, stride=stride, scope='%s/conv_shortcut'%(name))
        net = tf.concat([net, shortcut], axis=3)
        net = slim.conv2d(net, out_nc, 1, normalizer_fn=slim.batch_norm, stride=1, scope='%s/conv1x1'%(name))
    
    return net

# deeplabv3论文中出现的一个模块，似乎有效果
def atrous_spatial_pyramid_pooling(inputs, output_stride, nc=256, act='relu'):
    act_fn = activate(type_=act)
    with tf.variable_scope("spatial_pyramid_pooling"):
        if output_stride not in [8, 16]:
            raise ValueError('output_stride must be either 8 or 16.')

        atrous_rates = [6, 12, 18]
        if output_stride == 8:
            atrous_rates = [2*rate for rate in atrous_rates]

        with slim.arg_scope([slim.conv2d], 
                    activation_fn=act_fn,
                    weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                    weights_regularizer=slim.l2_regularizer(0.0005),
                    padding='SAME'):
            inputs_size = tf.shape(inputs)[1:3]
            conv_1x1 = slim.conv2d(inputs, nc, [1, 1], stride=1, scope="conv_1x1")
            conv_3x3_1 = slim.conv2d(inputs, nc, [3, 3], stride=1, rate=atrous_rates[0], scope='conv_3x3_1')
            conv_3x3_2 = slim.conv2d(inputs, nc, [3, 3], stride=1, rate=atrous_rates[1], scope='conv_3x3_2')
            conv_3x3_3 = slim.conv2d(inputs, nc, [3, 3], stride=1, rate=atrous_rates[2], scope='conv_3x3_3')

            with tf.variable_scope("image_level_features"):
                image_level_features = tf.reduce_mean(inputs, [1, 2], name='global_average_pooling', keepdims=True)
                image_level_features = slim.conv2d(image_level_features, nc, [1, 1], stride=1, scope='conv_1x1')
                image_level_features = ensure_type(image_level_features, tf.float32)
                image_level_features = tf.image.resize_bilinear(image_level_features, inputs_size, name='upsample')
                image_level_features = ensure_type(image_level_features, inputs.dtype)

            net = tf.concat([conv_1x1, conv_3x3_1, conv_3x3_2, conv_3x3_3, image_level_features], axis=3, name='concat')
            net = slim.conv2d(net, nc, [1, 1], stride=1, normalizer_fn=slim.batch_norm, scope='conv_1x1_concat')
            return net

def ConvBlock(net, in_nc, out_nc, stride=1, act='relu', middle_type='conv', name='ConvBlock'):
    act_fn = activate(type_=act)
    weights_init = tf.truncated_normal_initializer(stddev=0.01)
    with tf.variable_scope(name):
        with slim.arg_scope([slim.conv2d], 
                            activation_fn=act_fn,
                            weights_initializer=weights_init,
                            weights_regularizer=slim.l2_regularizer(0.0005),
                            padding='SAME'):
            net = slim.conv2d(net, in_nc, 1, stride=1)
            if middle_type == 'conv':
                net = slim.conv2d(net, in_nc, 3, normalizer_fn=slim.batch_norm, stride=stride)
            elif middle_type == 'depthwise':
                W = tf.get_variable('depthwise_weights', [3, 3, in_nc, 1], dtype=tf.float32, initializer=weights_init)
                net = tf.nn.depthwise_conv2d(net, W, [1, stride, stride, 1], padding='SAME', data_format='NHWC')
            net = slim.conv2d(net, out_nc, 1, stride=1)
            return net

# net: (N, H, W, C)
# out_nc: 输出的通道数
# Unet结构示例：
# downBlock(level=16)-downBlock(level=8)-downBlock(level=4)-downBlock(level=2)-downBlock(level=1)
# -upBlock(level=2)-upBlock(level=4)-upBlock(level=8)-upBlock(level=16)
# block_opt: block的一些参数
# kernel_size: conv大小
# normalize: 是否加BN
# act: 激活函数类型
# upsample_type: upsample类型
def Unet(net, out_nc, level, downblock_opt=None, upblock_opt=None, kernel_size=3, 
        normalize=True, act='relu', downsample_type='conv', upsample_type='bilinear',
        with_aspp=False):
    stride = 1+int(level!=1)
    norm = slim.batch_norm if normalize else None
    in_nc = out_nc//2
    inputs_size = tf.shape(net)[1:3]
    act_fn = activate(type_=act)

    down_nb = downblock_opt['nb']
    down_type = downblock_opt['type']
    up_nb = upblock_opt['nb']
    up_type = upblock_opt['type']
    weights_init = tf.truncated_normal_initializer(stddev=0.01)
    with slim.arg_scope([slim.conv2d], 
                    activation_fn=act_fn,
                    weights_initializer=weights_init,
                    weights_regularizer=slim.l2_regularizer(0.0005),
                    normalizer_fn=norm,
                    padding='SAME'):
        # downBlock: conv-block(支持conv、skconv和RDB三种)-conv
        if downsample_type == 'depthwise':
            net = slim.conv2d(net, in_nc, 1, stride=1)
        else:
            net = slim.conv2d(net, in_nc, kernel_size, stride=1)
        for i in range(down_nb):
            if down_type == 'conv':
                net = ConvBlock(net, in_nc, in_nc, act=act, stride=1, name="Down_Conv%d_%d"%(level, i))
            elif down_type == 'skconv':
                net = SKConv(net, in_nc, nconv=downblock_opt['nconv'], r=downblock_opt['r'], 
                            stride=1, L=downblock_opt['L'], act=act, name="Down_SKConv%d_%d"%(level, i), normalize=normalize)
            elif down_type == 'RDB':
                net = RDB(net, in_nc, nconv=downblock_opt['nconv'], gc=downblock_opt['gc'],  
                            stride=1, act=act, name='Down_RDB%d_%d'%(level, i))

        if downsample_type == 'conv':
            net = slim.conv2d(net, out_nc, 3, stride=stride)
        elif downsample_type == 'depthwise':
            W = tf.get_variable('Down_DepthWise_Weights%d'%(level), [3, 3, in_nc, 1], dtype=tf.float32, initializer=weights_init)
            net = tf.nn.depthwise_conv2d(net, W, [1, stride, stride, 1], padding='SAME', data_format='NHWC')
            net = slim.conv2d(net, out_nc, 1, stride=1)

        if level > 1:
            tmp = net
            # level>1时接着添加Unet
            net = Unet(net, out_nc*2, level//2, downblock_opt, upblock_opt, 
                        kernel_size, normalize, act, downsample_type, upsample_type,
                        with_aspp)
            # upBlock: concat-conv-block(支持conv、skconv和RDB三种)-upsample
            net = tf.concat([tmp, net], axis=3)
            net = slim.conv2d(net, out_nc, kernel_size, stride=1)
            for _ in range(up_nb):
                if up_type == 'conv':
                    net = ConvBlock(net, out_nc, out_nc, act=act, stride=1, name="Up_Conv%d_%d"%(level, i))
                elif up_type == 'skconv':
                    net = SKConv(net, out_nc, nconv=upblock_opt['nconv'], r=upblock_opt['r'], 
                            stride=1, L=upblock_opt['L'], act=act, name="Up_SKConv%d_%d"%(level, i), 
                            normalize=normalize)
                elif up_type == 'RDB':
                    net = RDB(net, out_nc, nconv=upblock_opt['nconv'], gc=upblock_opt['gc'],  
                                stride=1, act=act, name='Up_RDB%d_%d'%(level, i))
            net = UpSample(net, inputs_size, type_=upsample_type, nc=out_nc, name="Upsample%d"%(level))

        elif with_aspp:
            net = atrous_spatial_pyramid_pooling(net, 16, act=act)
        
        return net

def UnetPlus(net, out_nc, level, downblock_opt=None, upblock_opt=None, kernel_size=3, 
            normalize=True, act='relu', downsample_type='conv', upsample_type='bilinear'):
    norm = slim.batch_norm if normalize else None
    in_nc = out_nc//2
    act_fn = activate(type_=act)

    down_nb = downblock_opt['nb']
    down_type = downblock_opt['type']
    up_type = upblock_opt['type']
    weights_init = tf.truncated_normal_initializer(stddev=0.01)

    with slim.arg_scope([slim.conv2d], 
                    activation_fn=act_fn,
                    weights_initializer=weights_init,
                    weights_regularizer=slim.l2_regularizer(0.0005),
                    normalizer_fn=norm,
                    padding='SAME'):
        features = {}
        features[0] = [net]
        num = int(math.log2(level))            
        for k in range(1, num+1):
            now_level = 2**(k-1)
            net = slim.conv2d(net, in_nc*now_level, kernel_size, stride=1)

            for i in range(down_nb):
                if down_type == 'conv':
                    net = ConvBlock(net, in_nc*now_level, in_nc*now_level, act=act, stride=1, name="Down_Conv%d_%d"%(k, i))
                elif down_type == 'RDB':
                    net = RDB(net, in_nc*now_level, nconv=downblock_opt['nconv'], gc=downblock_opt['gc'],  
                            stride=1, act=act, name='Down_RDB%d_%d'%(k, i))
            
            if downsample_type == 'conv':
                net = slim.conv2d(net, out_nc*now_level, 3, stride=2)
            elif downsample_type == 'depthwise':
                W = tf.get_variable('Down_DepthWise_Weights%d'%(k), [3, 3, in_nc*now_level, 1], dtype=tf.float32, initializer=weights_init)
                net = tf.nn.depthwise_conv2d(net, W, [1, 2, 2, 1], padding='SAME', data_format='NHWC')
                net = slim.conv2d(net, out_nc*now_level, 1, stride=1)
            features[k] = [net]
        
        for k in range(num, 0, -1):
            now_level = 2**(k-1)
            tmp = features[k][0]
            inputs_size = tf.shape(tmp)[1:3]
            if up_type == 'conv':
                tmp2 = ConvBlock(tmp, out_nc*now_level, out_nc*now_level, act=act, stride=1, name="Up_Conv%d_0"%(k))
            elif up_type == 'RDB':
                tmp2 = RDB(tmp, out_nc*now_level, nconv=upblock_opt['nconv'], gc=upblock_opt['gc'],  
                            stride=1, act=act, name='Up_RDB%d_0'%(k))
            features[k-1].append(UpSample(tmp2, inputs_size*2, type_=upsample_type, nc=out_nc*now_level, name="Upsample%d_0"%(k)))

            for i in range(1, len(features[k])):
                tmp = tf.concat([tmp, features[k][i]], axis=3)
                if up_type == 'conv':
                    tmp2 = ConvBlock(tmp, out_nc*now_level, out_nc*now_level, act=act, stride=1, name="Up_Conv%d_%d"%(k, i))
                elif up_type == 'RDB':
                    tmp2 = slim.conv2d(tmp, out_nc*now_level, 1, stride=1)
                    tmp2 = RDB(tmp2, out_nc*now_level, nconv=upblock_opt['nconv'], gc=upblock_opt['gc'],  
                                stride=1, act=act, name='Up_RDB%d_%d'%(k, i))
                features[k-1].append(UpSample(tmp2, inputs_size*2, type_=upsample_type, nc=out_nc*now_level, name="Upsample%d_%d"%(k, i)))
        
        net = features[0][0]
        for i in range(1, len(features[0])):
            net = tf.concat([net, features[0][i]], axis=3)
        net = ConvBlock(net, out_nc*3, out_nc, act=act, stride=1, name="Last_Conv")
        
        return net

def VGG16(in_, name, reuse=False):
    act_fn = activate(type_='relu')
    with tf.variable_scope(name, reuse=reuse):
        with slim.arg_scope([slim.conv2d, slim.fully_connected], 
                        activation_fn=act_fn,
                        weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                        weights_regularizer=slim.l2_regularizer(0.0005)):
            with slim.arg_scope([slim.conv2d], padding='SAME'):
                net = slim.repeat(in_, 2, slim.conv2d, 64, [3, 3], scope='conv1')
                net = slim.max_pool2d(net, [2, 2], scope='pool1')
                net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
                net = slim.max_pool2d(net, [2, 2], scope='pool2')
                net = slim.repeat(net, 2, slim.conv2d, 256, [3, 3], scope='conv3')
                net = slim.max_pool2d(net, [2, 2], scope='pool3')
                net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
                net = slim.max_pool2d(net, [2, 2], scope='pool4')
                net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
                net = slim.max_pool2d(net, [2, 2], scope='pool5')

                net = slim.flatten(net, scope='flat5')
                net = slim.fully_connected(net, 4096, scope='fc6', activation_fn=None)

                return net