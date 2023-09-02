def reset_model_weights(layer):
    if hasattr(layer, 'reset_parameters'):
        print(f"weights were resetted for {layer}")
        layer.reset_parameters()
    else:
        if hasattr(layer, 'children'):
            for child in layer.children():
                reset_model_weights(child)