def cellboxes_to_device (out, S=7):
    converted_pred = convert_cellboxes(out).rehape(out.shape[0], S*S,-1)