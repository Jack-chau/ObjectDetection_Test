import torch

#   predictions (tensor): batch_size, S, S, B*5+C (B bounding_boxes in cell * 5=(iou,x,y,w,h) + classes_prob)
#   SxS grid cell in each imahes
#   batch_size is 64 from the paper
def convertCellBoxes(predictions,S=7):
    predictions = predictions.to('cpu')
    batch_size = predictions.shape[0]
    predictions.reshape = predictions.reshape(batch_size,7,7,30)
    bboxes1 = predictions[...,21:25] #... for batch_size, 0-19 for each class_prob, 20 for iou_prob, 21:25 for x,y,w,h
    bboxes2 = predictions[...,26:30] # box2 in each cell
    scores = torch.cat(
        (predictions[...,20].unsqueeze(0),predictions[...,25].unsqueeze(0)), dim=0
    ) # extract iou_prob from B and convert to 0 dimension
#   torch.unsqueeze(0) to add a dimenions at 0 dimenion, because torch need 2d dimension tensor
#   torch.cat((tuble),dimension) concat two tensor each other
    best_box = scores.argmax(0).unsqueeze(-1) # return the maximun value (index) in each score tensor, reshape into 2x1 tensor
    # elements rise , that's why boxes
    best_boxes = bboxes1 * (1 - best_box) + best_box * bboxes2
    '''
    Since B=2 (each cell we predict 2 boxes) and we extract the iou_prob 
    source (1d_tensor)= [iou_B1,iou_B2]
    best_box: best_box will return two values either 0 or 1 of the highest iou_prob's index
    best_box: if box_1 has higher iou than box_2 = [[0],[0]], else [[1],[0]]
    *** best_boxes:
        if (box_1 has the higher iou_proba than box_2):
            best_box = [[0],[0]]
            best_boxes = bboxes1 * (1 -0) + best_box * bboxes2
            best_boxes = bboxes1*1 + 0 * bboxes2
            best_boxes = [x,y,w,h] from bboxes_1
        elif (box_2 has higher iou_prob than box_1):
            best_box = [[1],[0]]
            best_boxes = bboxes_1 * (1-1) + 1 * bboxes_2 
            best_boxes = 0 + bboxes_2
    '''
    cell_indices = torch.arange(7).reshape(batch_size,7,1).unsqueeze(-1)
    """
    cell_indices (64x7x7) tensor:
        1. Create 1d tensor [0,1,2,3,4,5,6] # column only
        2. repeat this tensor 64 times with 7 rows
        3. turn out will be 64x7x7 shape 
    """
    x = 1 / S * (best_boxes[...,:1] + cell_indices)

    """
        best_boxes = [...,x,y,w,h] 
        # Since we split an image into 7x7 cells,
        In each cell, we predicted B bounding boxes and selected the one with the highest iou.
        Therefore we have 49 (7x7) cells and 49 bounding boxes.
        
        However, the bounding boxes gives the x and y representing the coordinate in
        that particular cell, w and h are the width and height response to the whole image.
        
        we have to scale the x and y coordinate in the whole 7x7 image, not the particular cell any more.
        
        For example:
            all predicted points are from the middle of the cell,
            0+0.5, 1+0.5,2+0.5..... is the coordinate response that image
            that's why (best_boxes[...,;1] + cell_indeces)
        Than:
               output * 1/7 for scale the image within 0-1.      
    """
    y = 1/7 * (best_boxes[...,1:2] + cell_indices.permute(0,2,1,3))
    w_y = best_boxes[...,2:4] * 1/7


























