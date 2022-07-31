import torch

def get_bboxes(
        loader,
        model,
        iou_threshold,
        pred_format = 'cells',
        box_format = 'midpoint',
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
):
    all_pred_boxes = []
    all_true_boxes = []

    # make sure model is in eval before get bboxes
    # turn model to evaluation mode
    model.eval()
    train_idx = 0

    for batch_idx, (x, labels) in enumerate(loader):
        x = x.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            predictions = model(x)

        batch_size = x.shape(0)
#...


