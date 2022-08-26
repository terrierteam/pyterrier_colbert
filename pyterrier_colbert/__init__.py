
#import importlib
#ranking = importlib.import_module('.' + 'indexing', package='pyterrier_colbert') 
#ranking = importlib.import_module('.' + 'ranking', package='pyterrier_colbert')


import torch
from colbert.utils.utils import print_message
from collections import OrderedDict, defaultdict
from colbert.modeling.colbert import ColBERT

DEFAULT_CLASS=ColBERT
DEFAULT_MODEL='bert-base-uncased'

def load_model(args, do_print=True, baseclass=DEFAULT_CLASS, basemodel=DEFAULT_MODEL):
    from colbert.parameters import DEVICE
    colbert = baseclass.from_pretrained(basemodel,
                                      query_maxlen=args.query_maxlen,
                                      doc_maxlen=args.doc_maxlen,
                                      dim=args.dim,
                                      similarity_metric=args.similarity,
                                      mask_punctuation=args.mask_punctuation)
    colbert = colbert.to(DEVICE)

    print_message("#> Loading model checkpoint.", condition=do_print)

    checkpoint = load_checkpoint(args.checkpoint, colbert, do_print=do_print)

    colbert.eval()

    return colbert, checkpoint

def load_colbert(args, do_print=True, baseclass=DEFAULT_CLASS, basemodel=DEFAULT_MODEL):
    from colbert.utils.runs import Run
    import ujson
    
    colbert, checkpoint = load_model(args, do_print, baseclass=baseclass, basemodel=basemodel)

    # TODO: If the parameters below were not specified on the command line, their *checkpoint* values should be used.
    # I.e., not their purely (i.e., training) default values.

    for k in ['query_maxlen', 'doc_maxlen', 'dim', 'similarity', 'amp']:
        if 'arguments' in checkpoint and hasattr(args, k):
            if k in checkpoint['arguments'] and checkpoint['arguments'][k] != getattr(args, k):
                a, b = checkpoint['arguments'][k], getattr(args, k)
                Run.warn(f"Got checkpoint['arguments']['{k}'] != args.{k} (i.e., {a} != {b})")

    if 'arguments' in checkpoint:
        if args.rank < 1:
            print(ujson.dumps(checkpoint['arguments'], indent=4))

    if do_print:
        print('\n')

    return colbert, checkpoint


def load_checkpoint(path, model, optimizer=None, do_print=True):
    if do_print:
        print_message("#> Loading checkpoint", path)

    if path.startswith("http:") or path.startswith("https:"):
        checkpoint = torch.hub.load_state_dict_from_url(path, map_location='cpu')
    else:
        checkpoint = torch.load(path, map_location='cpu')

    state_dict = checkpoint['model_state_dict']
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k
        if k[:7] == 'module.':
            name = k[7:]
        new_state_dict[name] = v

    checkpoint['model_state_dict'] = new_state_dict

    # check for transformer version mismatch and patch address accordingly
    import transformers
    from packaging import version
    strict = True
    if version.parse(transformers.__version__).major >= 4 and 'bert.embeddings.position_ids' not in checkpoint['model_state_dict']:
        strict = False
    
    model.load_state_dict(checkpoint['model_state_dict'], strict=strict)

    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if do_print:
        print_message("#> checkpoint['epoch'] =", checkpoint['epoch'])
        print_message("#> checkpoint['batch'] =", checkpoint['batch'])

    return checkpoint
