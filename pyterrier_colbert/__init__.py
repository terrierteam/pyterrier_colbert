
#import importlib
#ranking = importlib.import_module('.' + 'indexing', package='pyterrier_colbert') 
#ranking = importlib.import_module('.' + 'ranking', package='pyterrier_colbert')


import torch
from colbert.utils.utils import print_message
from collections import OrderedDict, defaultdict

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
