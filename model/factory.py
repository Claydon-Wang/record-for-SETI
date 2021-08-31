import logging
import torch.nn as nn
from .registry import is_model, is_model_in_modules, model_entrypoint
from .helpers import load_checkpoint
from .layers import set_layer_config
from .hub import load_model_config_from_hf


class BasicImageModel(nn.Module):
    def __init__(self, cfg):
        """Initialize"""
        super(BasicImageModel, self).__init__()
        in_channels = cfg.IN_CHANS
        if cfg.SELF_TAIL:
            self.tail = nn.Sequential(
                nn.Conv2d(in_channels, 32, 3, 2, 1),
                nn.BatchNorm2d(32),
                nn.SiLU(inplace=True),
                nn.Conv2d(32, 64, 3, 1, 1),
                nn.BatchNorm2d(64),
                nn.SiLU(inplace=True),
                nn.Conv2d(64, 64, 3, 1, 1),
                nn.BatchNorm2d(64),
                nn.SiLU(inplace=True),
            )
            in_channels = 64
        else:self.tail = None
        if cfg.SELF_HEAD:
            self.backbone = create_model(
                cfg.NAME, num_classes=0, pretrained=cfg.PRETRAIN, in_chans=in_channels)
            in_features = self.backbone.num_features
            logging.info(f"{cfg.NAME}: {in_features}")
            logging.info(f"load imagenet pretrained: {cfg.PRETRAIN}")
        
            # prepare head clasifier
            self.dims_head = cfg.NUM_CLASSES
            if self.dims_head[0] is None:
                self.dims_head[0] = in_features
                
            layers_list = []
            for i in range(len(self.dims_head) - 2):
                in_dim, out_dim = self.dims_head[i: i + 2]
                layers_list.extend([
                    nn.Linear(in_dim, out_dim),
                    nn.SiLU(inplace=True),
                    #nn.ReLU(inplace=True), 
                    nn.Dropout(cfg.DROPOUT),])
            layers_list.append(
                nn.Linear(self.dims_head[-2], self.dims_head[-1]))
            self.head_cls = nn.Sequential(*layers_list)
        else:
            self.backbone = create_model(
                cfg.NAME, num_classes=cfg.OUTPUT_DIM, pretrained=cfg.PRETRAIN, in_chans=in_channels)
            self.head_cls = None

    def forward(self, x):
        """Forward"""
        if self.tail is not None:
            x = self.tail(x)
        h = self.backbone(x)
        if self.head_cls is not None:
            h = self.head_cls(h)
        return h


def split_model_name(model_name):
    model_split = model_name.split(':', 1)
    if len(model_split) == 1:
        return '', model_split[0]
    else:
        source_name, model_name = model_split
        assert source_name in ('timm', 'hf_hub')
        return source_name, model_name


def safe_model_name(model_name, remove_source=True):
    def make_safe(name):
        return ''.join(c if c.isalnum() else '_' for c in name).rstrip('_')
    if remove_source:
        model_name = split_model_name(model_name)[-1]
    return make_safe(model_name)


def create_model(
        model_name,
        pretrained=False,
        checkpoint_path='',
        scriptable=None,
        exportable=None,
        no_jit=None,
        **kwargs):
    """Create a model

    Args:
        model_name (str): name of model to instantiate
        pretrained (bool): load pretrained ImageNet-1k weights if true
        checkpoint_path (str): path of checkpoint to load after model is initialized
        scriptable (bool): set layer config so that model is jit scriptable (not working for all models yet)
        exportable (bool): set layer config so that model is traceable / ONNX exportable (not fully impl/obeyed yet)
        no_jit (bool): set layer config so that model doesn't utilize jit scripted layers (so far activations only)

    Keyword Args:
        drop_rate (float): dropout rate for training (default: 0.0)
        global_pool (str): global pool type (default: 'avg')
        **: other kwargs are model specific
    """
    source_name, model_name = split_model_name(model_name)

    # Only EfficientNet and MobileNetV3 models have support for batchnorm params or drop_connect_rate passed as args
    is_efficientnet = is_model_in_modules(model_name, ['efficientnet', 'mobilenetv3'])
    if not is_efficientnet:
        kwargs.pop('bn_tf', None)
        kwargs.pop('bn_momentum', None)
        kwargs.pop('bn_eps', None)

    # handle backwards compat with drop_connect -> drop_path change
    drop_connect_rate = kwargs.pop('drop_connect_rate', None)
    if drop_connect_rate is not None and kwargs.get('drop_path_rate', None) is None:
        print("WARNING: 'drop_connect' as an argument is deprecated, please use 'drop_path'."
              " Setting drop_path to %f." % drop_connect_rate)
        kwargs['drop_path_rate'] = drop_connect_rate

    # Parameters that aren't supported by all models or are intended to only override model defaults if set
    # should default to None in command line args/cfg. Remove them if they are present and not set so that
    # non-supporting models don't break and default args remain in effect.
    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    if source_name == 'hf_hub':
        # For model names specified in the form `hf_hub:path/architecture_name#revision`,
        # load model weights + default_cfg from Hugging Face hub.
        hf_default_cfg, model_name = load_model_config_from_hf(model_name)
        kwargs['external_default_cfg'] = hf_default_cfg  # FIXME revamp default_cfg interface someday

    if is_model(model_name):
        create_fn = model_entrypoint(model_name)
    else:
        raise RuntimeError('Unknown model (%s)' % model_name)

    with set_layer_config(scriptable=scriptable, exportable=exportable, no_jit=no_jit):
        model = create_fn(pretrained=pretrained, **kwargs)

    if checkpoint_path:
        
        load_checkpoint(model, checkpoint_path)

    return model
