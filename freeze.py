
def freeze_model_weights(model, layers='0'):
    """
    freeze the model weights based on the layer names
    for example, if the layers contain '0', then if the name of the parameter contains _FREEZE_KEY['0'] will be frozen
    """
    print('Going to apply weight frozen')
    assert len(set(layers) - {'0', '1', '2', '3', '4'})==0, "right now only support freeze the combination of 0,1,2,3,4 layers" 
    print('before frozen, require grad parameter names:')

    _FREEZE_KEY = {'0': ['ginlayers.0','linears_prediction_classification.0'],
               '1': ['ginlayers.1','linears_prediction_classification.1'],
               '2': ['ginlayers.2','linears_prediction_classification.2'],
               '3': ['ginlayers.3','linears_prediction_classification.3'],
               '4': ['ginlayers.4','linears_prediction_classification.4'],
               }
    for name, param in model.named_parameters():
        if param.requires_grad:print(name)
    freeze_keys = []
    for layer in layers:
        freeze_keys += _FREEZE_KEY[layer]
    print('freeze_keys', freeze_keys)
    for name, param in model.named_parameters():
        if param.requires_grad and any(key in name for key in freeze_keys):
            param.requires_grad = False
    print('after frozen, require grad parameter names:')
    for name, param in model.named_parameters():
        if param.requires_grad:print(name)
    return model

def unfreeze_model_weights(model):
    """
    unfreeze the model weights
    """
    print('Going to apply weight UNfrozen')
    print('before unfrozen, require grad parameter names:')
    for name, param in model.named_parameters():
        if param.requires_grad:print(name)
    print('after unfrozen, require grad parameter names:')
    for name, param in model.named_parameters():
        param.requires_grad = True
        if param.requires_grad:print(name)
    return model  