
import os
from collections import OrderedDict
from datetime import datetime

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.models import clone_model
# import tensorflow.compat.v1 as tf
import tensorflow as tf



import os
import logging
import pickle
import json
import tensorflow as tf
from packaging import version

if version.parse(tf.__version__) >= version.parse('2.0.0'):
    import tensorflow.keras.backend as K
    from tensorflow.keras.models import Model, model_from_config
    from tensorflow.keras.layers import Input, Permute, Softmax
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.regularizers import l2
else:
    import keras.backend as K
    from keras.models import Model, model_from_config
    from keras.layers import Input, Permute, Softmax
    from keras.optimizers import Adam
    from keras.regularizers import l2

from omegaconf import DictConfig
from typing import Callable, List
from vbg.blocks import ResidualDepthwiseBlock, DownBlock, BottleneckBlock, Upsample2xBlock, ASPPv3Block, HeadBlock
from vbg.losses import Losses
from vbg.metrics import get_test_time_metrics, get_train_time_metrics
from utils.artifact import WandbArtifact
from utils.params_utils import params_to_list, expand_username, is_url
from utils.preprocessing_utils import default_preproc
from utils.tf_compatibility import detect_and_fix_version, setup_float32_policy
from utils.path_utils import Paths
from utils import callbacks


log = logging.getLogger(name=__name__)


class JPU_Deci(object):
    '''
    The SegModel object contains methods to build and train VBG Model

    Args:
        model_config (dict): model config with num of layers / filters / dilation pattern / etc.
        weights_config (dict): config of model weights
        mode (str): "train" (include supervision if exists) or "test" (remove supervision, take only one output)
    '''

    def __init__(self, mode: str, model_config: dict):
        if mode not in ["train", "test"]:
            raise ValueError("Unexpected value for 'mode' parameter, expected values: 'train', 'test'")
        self.mode = mode
        self.model_config = model_config
        self.model = None

    @classmethod
    def new(cls,
            model_config: dict,
            mode: str = "test"):
        log.info("No weights")

        obj = cls(mode=mode, model_config=model_config)
        obj.model = obj.build_model(model_config=model_config)

        return obj

    @classmethod
    def load(cls,
             model_path: str,
             mode: str = "test"):

        if not model_path:
            raise ValueError('Cannot load model: weights file is absent!')
        elif is_url(model_path):
            # Auto load weights file if it is a link
            model_path = str(WandbArtifact.download(url=model_path).path)

        obj = cls(mode=mode, model_config=None)
        with expand_username(model_path).open("rb") as file:
            obj.backup = pickle.load(file)
            obj.backup['model'] = json.loads(obj.backup['model'])

        if mode == "test":
            setup_float32_policy(model_json=obj.backup['model'])
        detect_and_fix_version(obj.backup['model'])

        obj.model = model_from_config(obj.backup['model'])
        obj.set_weights(obj.backup["weights"])
        log.info(f"Weights from {model_path} are loaded successfully")
        return obj

    def set_weights(self, weights: dict):
        for layer in self.model.layers:
            if "permute" not in layer.name:
                layer.set_weights(weights[layer.name])

    def build_model(self, model_config, coreml_compatible: bool = False):
        # filters = params_to_list(model_config.filters)
        # layers_num = params_to_list(model_config.layers_num)
        # aspp_dilations = params_to_list(model_config.aspp.dilations)
        #
        # l2_regularization_factor = model_config.l2_regularization_factor
        # kernel_regularizer = l2(l2_regularization_factor) if l2_regularization_factor else None
        #
        # shape_size = (model_config.input_shape.height, model_config.input_shape.width, 4)
        # if K.image_data_format() == "channels_first":
        #     shape_size = (4, model_config.input_shape.height, model_config.input_shape.width)
        #
        # input_layer = Input(shape=shape_size, name="input")
        #
        # supervision = []
        #
        # # Encoder stages
        # x = DownBlock(x=input_layer, filters=filters[0], kernel=(7, 7))
        # for i in range(layers_num[0]):
        #     x = BottleneckBlock(x=x, filters=filters[0], addition=True, kernel_regularizer=kernel_regularizer)
        #
        # encoder_stage_1 = x
        #
        # x = DownBlock(x=x, filters=filters[1])
        # for i in range(layers_num[1]):
        #     x = ResidualDepthwiseBlock(x=x, filters=filters[1], dilation=(1, 1), kernel_regularizer=kernel_regularizer)
        #
        # encoder_stage_2 = x
        #
        # x = DownBlock(x=x, filters=filters[2])
        # for i in range(layers_num[2]):
        #     x = ResidualDepthwiseBlock(x=x, filters=filters[2], dilation=(1, 1), kernel_regularizer=kernel_regularizer)
        #
        # encoder_stage_3 = x
        #
        # x = DownBlock(x=x, filters=filters[3])
        # for i in range(layers_num[3]):
        #     x = ResidualDepthwiseBlock(x=x, filters=filters[3], dilation=(1, 1), kernel_regularizer=kernel_regularizer)
        #
        # if model_config.aspp.use:
        #     x = ASPPv3Block(x=x, filters=32, dilation_rates=aspp_dilations)
        #
        # # Decoder stages
        # x = HeadBlock(low_resolution_layer=x, high_resolution_layer=encoder_stage_3,
        #               high_resolution_filters=filters[2], output_filters=filters[2], last_relu=True)
        #
        # x = HeadBlock(low_resolution_layer=x, high_resolution_layer=encoder_stage_2,
        #               high_resolution_filters=filters[1], output_filters=filters[1], last_relu=True)
        #
        # x = HeadBlock(low_resolution_layer=x, high_resolution_layer=encoder_stage_1,
        #               high_resolution_filters=filters[0], output_filters=filters[0], last_relu=True)
        #
        # x = Upsample2xBlock(x=x, filters=2, kernel_regularizer=kernel_regularizer)
        #
        # # Pass dtype explicitly to be sure output layers correct with mixed precision
        # if coreml_compatible:
        #     out = Softmax(axis=1, dtype='float32', name="output")(x)
        # elif K.image_data_format() == "channels_first":
        #     x = Permute((2, 3, 1))(x)
        #     x = Softmax(axis=-1, dtype='float32')(x)
        #     out = Permute((3, 1, 2), dtype='float32', name="output")(x)
        # else:
        #     out = Softmax(axis=-1, dtype='float32', name="output")(x)
        #
        # outputs = out
        # if self.mode == "train" and supervision:
        #     outputs = [out] + supervision
        #
        # model = Model(inputs=input_layer, outputs=outputs)
        #
        # return model

        model = build_model_deci_jpu((128,224,4))
        return model

    def get_channels_first_model(self, model_config: dict, coreml_compatible: bool = False):
        '''Convert to channels first model'''
        K.set_image_data_format("channels_first")
        ch_f_model = self.build_model(model_config=model_config,
                                      coreml_compatible=coreml_compatible)

        # channels-first model can contain extra layers (such as 'Permute'), just skip this layers while set weights
        skip_layers = {"permute"}
        channels_last_layer_index = 0
        for channels_first_layer in ch_f_model.layers:
            is_skip = any(skip_layer in channels_first_layer.name for skip_layer in skip_layers)
            is_last_layer = self.model.layers[channels_last_layer_index].name == "output"
            if is_skip or is_last_layer:
                continue
            channels_last_weights = self.model.layers[channels_last_layer_index].get_weights()
            channels_first_layer.set_weights(channels_last_weights)
            channels_last_layer_index += 1

        return ch_f_model

    def predict(self, data):
        normilized_data = default_preproc(data)
        return self.model.predict(normilized_data)

    def train(self, config: DictConfig,
              train_data_generator: Callable,
              valid_data_generator: Callable,
              test_generators: List[Callable],
              temporal_generators: List[Callable]) -> list:
        optimizer = Adam(lr=config.train.start_lr)

        loss = config.train.loss
        metrics = params_to_list(config.train.metrics)
        postpone_metrics = params_to_list(config.train.postpone_metrics)
        temporal_metrics = params_to_list(config.train.temporal_metrics)

        available_train_time_metrics = get_train_time_metrics()
        all_available_metrics = {**get_test_time_metrics(), **get_train_time_metrics()}

        assert loss in Losses, f"Unknown loss {loss}, expected values: {', '.join(Losses.keys())}"
        for metric in metrics:
            assert metric in available_train_time_metrics, f"Unknown metric {metric}, "\
                                                           f"expected values: {', '.join(available_train_time_metrics.keys())}"

        for metric in temporal_metrics + postpone_metrics:
            assert metric in all_available_metrics, f"Unknown metric {metric}, "\
                                                    f"expected values: {', '.join(all_available_metrics.keys())}"

        self.model.compile(optimizer=optimizer,
                           loss=[Losses[config.train.loss]],
                           metrics=[available_train_time_metrics[metric] for metric in metrics])

        callbacks_to_save = {}
        callbacks_list = []

        if config.train.lr_decay.use:
            reduce_lr = callbacks.CustomReduceLROnPlateau(factor=config.train.lr_decay.scheduler.factor,
                                                          patience=config.train.lr_decay.scheduler.patience,
                                                          min_lr=config.train.lr_decay.scheduler.min_lr,
                                                          verbose=1)
            callbacks_to_save["reduce_lr"] = reduce_lr
            callbacks_list.append(reduce_lr)

        callbacks_list.append(callbacks.ExtendedTqdm())

        if os.environ.get('OFFLINE_MODE', 'false') == 'true':
            callbacks_list.append(callbacks.SaveModel(callbacks_to_save=callbacks_to_save,
                                                      model_config=self.model_config,
                                                      monitor='val_mean_iou', mode='max',))
            TBCallback = callbacks.TensorBoard
        else:
            callbacks_list.append(callbacks.ClearML())
            callbacks_list.append(callbacks.EpochTime())
            callbacks_list.append(callbacks.BatchTime(batch_size=config.train.batch_size))
            callbacks_list.append(callbacks.BatchDelayTime())

            callbacks_list.append(callbacks.PostponeTemporal(data_generators=temporal_generators,
                                                             metrics=[all_available_metrics[metric]
                                                                      for metric in temporal_metrics],
                                                             evaluate_only_best=True,
                                                             monitor='val_mean_iou',
                                                             mode='max'))

            callbacks_list.append(callbacks.Postpone(data_generators=test_generators,
                                                     metrics=[all_available_metrics[metric]
                                                              for metric in postpone_metrics],
                                                     monitor='val_mean_iou',
                                                     mode='max'))

            callbacks_list.append(callbacks.CustomWandb(callbacks_to_save=callbacks_to_save,
                                                        model_config=self.model_config,
                                                        monitor='val_mean_iou',
                                                        mode='max',
                                                        log_best_prefix='best_'))

            TBCallback = callbacks.CustomTensorboard

        if config.train.tensorboard.use:
            tensorboard_log_dir = Paths.output() / "tensorboard"
            log.debug(f"Using {tensorboard_log_dir} to save tensorboard logs")
            callbacks_list.append(TBCallback(log_dir=tensorboard_log_dir,
                                             update_freq=config.train.tensorboard.update_freq))
        log.info("Start training...")
        history = self.model.fit(train_data_generator,
                                 verbose=0,
                                 epochs=config.train.epochs,
                                 callbacks=callbacks_list,
                                 validation_data=valid_data_generator,
                                 initial_epoch=config.train.initial_epoch - 1,
                                 shuffle=False,
                                 max_queue_size=config.train.max_queue_size,
                                 use_multiprocessing=config.train.use_multiprocessing,
                                 workers=config.train.num_of_workers)
        log.info("Training is done")

        return history





def down_block(x, filters):
    x = Conv2D(filters, (3, 3), strides=(2, 2), padding='same')(x)
    return x


def up_block(x, filters):
    x = Conv2DTranspose(filters, (3, 3), strides=(2, 2), padding='same')(x)
    return x


def res_block(x, filters, kernel=(3, 3), strides=(1, 1), dilation=(1, 1), name=''):
    res = x
    x = SeparableConv2D(filters=filters, kernel_size=kernel, strides=strides, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = SeparableConv2D(filters=filters, kernel_size=kernel, dilation_rate=dilation, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    return Add()([x, res])


def proper_pixel_shuffle(x, scale):
    return Lambda(lambda x: tf.nn.depth_to_space(x, scale))


def build_model_deci_jpu(input_shape):
    inp = Input(shape=input_shape, name="input")

    ### ENCODER

    # STAGE 1
    x = down_block(inp, 16)
    for i in range(1):
        x = res_block(x, 16, name='stage1')
    skip_1 = x  # 72, 128, 16

    # STAGE 2
    x = down_block(x, 32)
    for i in range(4):
        x = res_block(x, 32)
    x = res_block(x, 32, name='stage2')
    skip_2 = x  # 36, 64, 32

    # STAGE 3
    x = down_block(x, 64)
    for i in range(4):
        x = res_block(x, 64)
    x = res_block(x, 64, name='stage3')
    skip_3 = x  # 18, 32, 64

    # STAGE 4
    x = down_block(x, 128)
    for i in range(1):
        x = res_block(x, 128)
    x = res_block(x, 128, name='stage4')
    skip_4 = x  # 9, 16, 128

    ### JPU
    endpoints = [skip_2, skip_3, skip_4]
    _, x = JPU_Deci_model(endpoints)
    print('JPU OUTPUT:', x.shape)  # --> JPU OUTPUT: (None, 18, 32, 64)

    ### DECODER
    x = up_block(x, 32)  # (None, 18, 32, 64) -> (None, 36, 64, 32)

    x = Add()([x, skip_2])

    for i in range(2):
        x = res_block(x, 32)

    pixel_shuffle = proper_pixel_shuffle(x, 2)
    x = pixel_shuffle(x)
    print('PS OUTPUT:', x.shape)  # (None, 36, 64, 32) -> (None, 72, 128, 8)

    pixel_shuffle = proper_pixel_shuffle(x, 2)
    x = pixel_shuffle(x)
    print('PS OUTPUT:', x.shape)  # (None, 72, 128, 8) -> (None, 144, 256, 2)

    out = Activation("softmax", name="output", dtype="float32")(x)

    model = Model(inputs=inp, outputs=out)

    return model



def JPU_Deci_model(endpoints: list, out_channels=16):
    def conv_block(tensor, num_filters, kernel_size, padding='same', strides=1, dilation_rate=1, w_init='he_normal'):
        x = tf.keras.layers.Conv2D(filters=num_filters,
                                   kernel_size=kernel_size,
                                   padding=padding,
                                   strides=strides,
                                   dilation_rate=dilation_rate,
                                   kernel_initializer=w_init,
                                   use_bias=False)(tensor)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        return x

    def sepconv_block(tensor, num_filters, kernel_size, padding='same', strides=1, dilation_rate=1, w_init='he_normal'):
        x = tf.keras.layers.SeparableConv2D(filters=num_filters,
                                            depth_multiplier=1,
                                            kernel_size=kernel_size,
                                            padding=padding,
                                            strides=strides,
                                            dilation_rate=dilation_rate,
                                            depthwise_initializer=w_init,
                                            use_bias=False)(tensor)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        return x

    h, w = endpoints[1].shape.as_list()[1:3]  # -> h=36, w=64, c=32

    for i in range(1, len(endpoints)):
        endpoints[i] = conv_block(tensor=endpoints[i], num_filters=out_channels, kernel_size=3)
        if i != 1:
            h_t, w_t = endpoints[i].shape.as_list()[1:3]
            scale = (h // h_t, w // w_t)
            endpoints[i] = tf.keras.layers.UpSampling2D(size=scale)(endpoints[i])

    yc = tf.keras.layers.Concatenate(axis=-1)(endpoints[1:])
    ym = []
    for rate in [1, 2, 4, 8]:
        ym.append(sepconv_block(yc, out_channels, 3, dilation_rate=rate))
    y = tf.keras.layers.Concatenate(axis=-1)(ym)
    return endpoints, y

#
# def model_weight_ensemble(members, weights):
#     # determine how many layers need to be averaged
#     n_layers = len(members[0].get_weights())
#     # create an set of average model weights
#     avg_model_weights = list()
#     for layer in range(n_layers):
#         # collect this layer from each model
#         layer_weights = np.array([model.get_weights()[layer] for model in members])
#         # weighted average of weights for this layer
#         avg_layer_weights = np.average(layer_weights, axis=0, weights=weights)
#         # store average layer weights
#         avg_model_weights.append(avg_layer_weights)
#     # create a new model with the same structure
#     model = clone_model(members[0])
#     # set the weights in the new
#     model.set_weights(avg_model_weights)
#     metrics = {
#         "output": iou_metric
#     }
#     losses = {
#         "output": combined_loss
#     }
#     model.compile(optimizer=Adam(lr=1e-3), loss=losses, metrics=metrics)
#     return model