import torch
from IoU import intersetion_over_union

def nms(bboxes,
        threshold,
        iou_threshold,
        box_format='corners',
):
#    prediction = [[1,0.9,x1,y1,x2,y2]] the structure follow by classes, probability of that classes, and the corresponding coordinant
    assert type(bboxes) == list
    bboxes = [box for box in bboxes if box[1] > threshold]
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True) # short for the highest probability
    bboxes_after_nms = []

    while bboxes:
        chosen_box = bboxes.pop(0)
        # list comprehension
        bboxes = [box
                  for box in bboxes
                  if box[0] != chosen_box[0]
                  or intersetion_over_union(
                        torch.tensor(chosen_box[2:]),
                        torch.tensor(box[2:]),
                        box_format = box_format) < iou_threshold]
        bboxes_after_nms.append(chosen_box)

        return bboxes_after_nms
