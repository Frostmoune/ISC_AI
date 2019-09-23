import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import variable_scope
from .common_helpers import ensure_type
from .module import *
from .loss import *
import horovod.tensorflow as hvd
from tensorflow.contrib.slim.nets import resnet_v2

# Encoder-Decoder架构的Encoder
def createEncoder(in_, opt, name):
    net = in_
    with variable_scope.variable_scope('%s/Encoder'%(name), 'Encoder', [net]):
        strides = opt['strides'] # 每一个block的stride
        kernel_size = opt['kernel_size'] # conv的大小
        nbs = opt['nbs'] # 每一块的block数量
        nconv = opt['nconv'] # block的nconv参数
        type_ = opt['type'] # block的类型
        act = opt['act'] # 激活函数类型
        with_atrous = opt['with_atrous'] # 是否加上论文中的atrous模块

        features = []
        act_fn = activate(type_=act)
        with slim.arg_scope([slim.conv2d], 
                    activation_fn=act_fn,
                    weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                    weights_regularizer=slim.l2_regularizer(0.0005),
                    padding='SAME'):

            net = slim.conv2d(net, 64, kernel_size, stride=1, scope='conv_start')
            features.append(net) # feature1x: 未downsample的feature
            
            for i, nb in enumerate(nbs):
                in_nc = 64*(2**i)
                if type_ == 'RRDB':
                    net = RRDB(net, in_nc, nb, nconv=nconv, gc=opt['gc'], kernel_size=opt['kernel_size'], 
                                stride=strides[i], name='RRDB%d'%(i), act=act)
                if type_ == 'SKNet':
                    net = SKBlock(net, in_nc, nb, nconv=nconv, stride=strides[i], r=opt['r'], 
                                L=opt['L'], name='SKBlock%d'%(i), act=act)
                if type_ == 'SKDense':
                    net = SKDense(net, in_nc, nb, nd=opt['nd'], nconv=nconv, gc=opt['gc'], act=act,
                                stride=strides[i], r=opt['r'], L=opt['L'], name='SKDense%d'%(i))
                if strides[i] == 2 and i < len(nbs) - 1:
                    features.append(net)
            
            features = features[::-1]
            if with_atrous:
                net = atrous_spatial_pyramid_pooling(net, 2**len(nbs), act=act)
            return net, features

# Encoder-Decoder架构的Decoder
def createDecoder(in_, net, features, opt, name):
    with variable_scope.variable_scope('%s/Decoder'%(name), 'Decoder', [net]):
        type_ = opt['type'] # block类型
        kernel_size = opt['kernel_size'] # conv大小
        nbs = opt['nbs'] # 每一部分的block数量
        concats = opt['concats'] # 是否与Encoder得到的特征concat
        upsample_type = opt['upsample_type'] # upsample类型
        act = opt['act'] # 激活函数类型
        num_classes = 3
        inputs_size = tf.shape(in_)[1:3]

        act_fn = activate(type_=act)
        with slim.arg_scope([slim.conv2d], 
                    activation_fn=act_fn,
                    weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                    weights_regularizer=slim.l2_regularizer(0.0005),
                    padding='SAME'):
            net = ensure_type(net, tf.float32)
            length = len(nbs)
            for k in range(length):
                level = 2**(length-k-1)
                net = UpSampleBlock(net, inputs_size//level, 1, type_=upsample_type, name='Upsample%d'%(k+1))
                # 可选择是否concat
                if concats[k]:
                    net = tf.concat([net, features[k]], axis=3, name='concat%d'%(k+1))

                if type_ == 'RDB' and nbs[k] > 0:
                    net = RRDB(net, 256, nbs[k], nconv=opt['nconv'], gc=opt['gc'], act=act,
                            kernel_size=kernel_size, stride=1, name='RRDB%d'%(k))
                elif type_ == 'Conv':
                    for i in range(nbs[k]):
                        net = slim.conv2d(net, 256, kernel_size, normalizer_fn=slim.batch_norm, 
                                        stride=1, scope='conv%d_%d'%(k+1, i))
                elif type_ == 'SKConv':
                    for i in range(nbs[k]):
                        net = SKConv(net, 256, nconv=opt['nconv'], r=opt['r'], act=act, 
                                    stride=1, L=opt['L'], name='SKConv%d_%d'%(k+1, i))
                net = ensure_type(net, features[k].dtype)
            
            net = slim.conv2d(net, 64, kernel_size=kernel_size, normalizer_fn=slim.batch_norm, 
                            stride=1, scope='conv%d'%(length+1))

            # 下面的部分是deeplabv3模型中本来就有的，这里未做更改
            logits = slim.conv2d(net, num_classes, [1, 1], stride=1, activation_fn=None, 
                                normalizer_fn=None, scope='conv_last')
            logits = ensure_type(logits, features[2].dtype)
            sm_logits = tf.nn.softmax(logits, axis=-1)
            return logits, sm_logits

# 未完成
def createFPN(in_, opt, name):
    net = in_
    with variable_scope.variable_scope(name, 'FPN', [net]):
        upsample_type = opt['upsample_type']
        act = opt['act']
        kernel_size = opt['kernel_size']
        nbs = opt['nbs']
        nconv = opt['nconv']
        block_type = opt['block_type']
        concat_type = opt['concat_type']
        num_classes = 3
        inputs_size = tf.shape(in_)[1:3]

        act_fn = activate(type_=act)
        with slim.arg_scope([slim.conv2d], 
                    activation_fn=act_fn,
                    weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                    weights_regularizer=slim.l2_regularizer(0.0005),
                    padding='SAME'):
            net = slim.conv2d(net, 64, 3, stride=1, scope="conv_0")
            features = []
            features.append(slim.conv2d(in_, 64, 1, stride=1))
            for i, nb in enumerate(nbs):
                level = 64*(2**i)
                if block_type == 'RRDB':
                    net = RRDB(net, level, nb, nconv=nconv, act=act, 
                                kernel_size=kernel_size, stride=2, name="RRDB%d"%(i))
                features.append(slim.conv2d(net, level, 1, stride=1))
            features = features[::-1]

            length = len(nbs)
            net = UpSampleBlock(features[0], inputs_size, length, type_=upsample_type, act=act, name="Upsample0")
            for i in range(1, length):
                level = length-i
                now_size = inputs_size//(2**level)
                if concat_type == 'concat':
                    features[i] = tf.concat([UpSampleBlock(features[i-1], now_size, 1, type_=upsample_type, 
                                            act=act, name="Upsample2x_%d"%(i)), features[i]], axis=3)
                    features[i] = slim.conv2d(features[i], 256, 1, stride=1, normalizer_fn=slim.batch_norm)
                elif concat_type == 'add':
                    features[i] = UpSampleBlock(features[i-1], now_size, 1, type_=upsample_type, 
                                act=act, name="Upsample2x_%d"%(i)) + features[i-1]
                net = tf.concat([net, UpSampleBlock(features[i], inputs_size, level, type_=upsample_type, 
                                act=act, name="Upsample%d"%(i))], axis=3)
            net = slim.conv2d(net, 256, 1, normalizer_fn=slim.batch_norm, stride=1)

            for i in range(3):
                net = slim.conv2d(net, 128//(2**i), 3, stride=1)
            logits = slim.conv2d(net, num_classes, 1, stride=1, activation_fn=None, normalizer_fn=None)
            logits = ensure_type(logits, in_.dtype)
            sm_logits = tf.nn.softmax(logits, axis=-1)
            return logits, sm_logits

# Unet架构
def createUnet(in_, opt, name):
    net = in_
    with variable_scope.variable_scope(name, 'Unet', [net]):
        kernel_size = opt['kernel_size'] # conv大小
        downsample_type = opt['downsample_type'] # downsample类型
        upsample_type = opt['upsample_type'] # upsample类型
        normalize = opt['normalize'] # 是否加BN层
        with_aspp = opt['with_aspp'] # 是否加aspp
        act = opt['act'] # 激活函数类型
        level = opt['level'] # 见module.Unet中的参数介绍
        out_nc = opt['nc'] # 见module.Unet中的参数介绍
        num_classes = 3
        downblock_opt = opt['down_block']
        upblock_opt = opt['up_block']

        act_fn = activate(type_=act)
        with slim.arg_scope([slim.conv2d], 
                    activation_fn=act_fn,
                    weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                    weights_regularizer=slim.l2_regularizer(0.0005),
                    padding='SAME'):
            net = Unet(net, out_nc, level, downblock_opt=downblock_opt, upblock_opt=upblock_opt, 
                        kernel_size=kernel_size, normalize=normalize, act=act, with_aspp=with_aspp,
                        downsample_type=downsample_type, upsample_type=upsample_type)
            net = slim.conv2d(net, out_nc, 3, stride=1, normalizer_fn=slim.batch_norm)

            # 下面的部分是deeplabv3模型中本来就有的，这里未做更改
            logits = slim.conv2d(net, num_classes, 1, stride=1, activation_fn=None, normalizer_fn=None)
            logits = ensure_type(logits, in_.dtype)
            sm_logits = tf.nn.softmax(logits, axis=-1)
            return logits, sm_logits

# Unet架构
def createUnetPlus(in_, opt, name):
    net = in_
    with variable_scope.variable_scope(name, 'UnetPlus', [net]):
        kernel_size = opt['kernel_size']
        downsample_type = opt['downsample_type']
        upsample_type = opt['upsample_type']
        normalize = opt['normalize']
        act = opt['act']
        level = opt['level']
        out_nc = opt['nc']
        num_classes = 3
        downblock_opt = opt['down_block']
        upblock_opt = opt['up_block']

        act_fn = activate(type_=act)
        with slim.arg_scope([slim.conv2d], 
                    activation_fn=act_fn,
                    weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                    weights_regularizer=slim.l2_regularizer(0.0005),
                    padding='SAME'):
            net = UnetPlus(net, out_nc, level, downblock_opt=downblock_opt, upblock_opt=upblock_opt, 
                        kernel_size=kernel_size, normalize=normalize, act=act, downsample_type=downsample_type, 
                        upsample_type=upsample_type)
            net = slim.conv2d(net, out_nc, 3, stride=1, normalizer_fn=slim.batch_norm)

            logits = slim.conv2d(net, num_classes, 1, stride=1, activation_fn=None, normalizer_fn=None)
            logits = ensure_type(logits, in_.dtype)
            sm_logits = tf.nn.softmax(logits, axis=-1)
            return logits, sm_logits

def createDiscriminator(opt, pred, label, name):
    fake, real = None, None
    if opt['type'] == 'VGG16':
        fake = VGG16(pred, name="%s/Discriminator_VGG16"%(name), reuse=False)
        real = VGG16(label, name="%s/Discriminator_VGG16"%(name), reuse=True)
    elif opt['type'] == 'ResNet101':
        model = resnet_v2.resnet_v2_101
        fake, _  = model(pred, num_classes=None, is_training=True)
        real, _  = model(pred, num_classes=None, is_training=True, reuse=True)
    return fake, real

def create(next_elem, opt):
    name = opt['Name']
    type_ = opt['Type']
    logits, sm_logits = None, None

    # Encoder-Decoder架构
    if type_ == 'Encoder-Decoder':
        net, features = createEncoder(next_elem[0], opt['Encoder'], name)
        logits, sm_logits = createDecoder(next_elem[0], net, features, opt['Decoder'], name)

    # Unet架构
    elif type_ == 'Unet':
        logits, sm_logits = createUnet(next_elem[0], opt['Generator'], name)
    
    # UnetPlus架构
    elif type_ == 'UnetPlus':
        logits, sm_logits = createUnetPlus(next_elem[0], opt['Generator'], name)

    # TODO
    elif type_ == 'FPN':
        logits, sm_logits = createFPN(next_elem[0], opt['Generator'], name)
    
    fake, real = None, None
    if 'Discriminator' in opt:
        predict = ensure_type(tf.expand_dims(tf.argmax(sm_logits, axis=3), -1), tf.float32)
        label = ensure_type(tf.expand_dims(next_elem[1], -1), tf.float32)
        fake, real = createDiscriminator(opt['Discriminator'], predict, label, name)

    return logits, sm_logits, fake, real