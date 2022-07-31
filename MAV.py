import torch
from collections import Counter
from IoU import intersetion_over_union

def mean_average_precision(
        pred_boxes, # all prediction boxes across all training example
        true_boxes, # true bboxes
        iou_threshold = 0.5,
        box_format = 'corners',
        num_classes = 20
):
    # pred_boxes (list): [[train_index, class_pred, prob_score,x1,y1,x2,y2],...]
    # train_index: what is the box from (which train image)
    # class_pred: what class is it

    average_precisions = [] # add the averager precision for each class in this class
    epsilon = 1e-6

    for c in range(num_classes):
        detections = []
        ground_truths = []

        for detection in pred_boxes:
            if detection[1] == c:
                detections.append(detection)

        for true_box in true_boxes:
            if true_box[1] == c:
                ground_truths.append(true_box)

        # Return a dictionary
        # let say: img 0 has 3 bboxes
        #            img 1 has 5 bboxes
        #amount_bboxes = {0:3,1:5}
        amount_bboxes = Counter([gt[0] for gt in ground_truths])

        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)
            # amount_bboxes = {0:torch.tensor([0,0,0]), 1:torch.tensor([0,0,0,0,0])}
            # doing this is because we want to find TP or FP amount all bboxes
        detections.sort(key=lambda x: x[2], reverse=True) # highest probability
        TP = torch.zeros((len(detections)))
        FP = torch.zeros((len(detections)))
        total_true_bboxes = len(ground_truths)

        for detection_idx, detection in enumerate(detections):
            ground_truth_img = [
                bbox for bbox in ground_truths if bbox[0] == detection[0]]

            num_gts = len(ground_truth_img)
            best_iou = 0

            for idx, gt in enumerate(ground_truth_img):
                iou = intersetion_over_union(
                        torch.tensor(detection[3:]),
                        torch.tensor(gt[3:]),
                        box_format = box_format
                )

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            if best_iou > iou_threshold:
                if amount_bboxes[detection[0]][best_gt_idx] == 0:
                    TP[detection_idx] = 1
                    amount_bboxes[detection[0]][best_gt_idx] = 1
                else:
                    FP[detection_idx] = 1

            else:
                FP[detection_idx] = 1
        # [1,1,0,1,0] -> [1,2,2,3,3]
        TP_cumsum = torch.cumsum(TP,dim=0)
        FP_cumsum = torch.cumsum(FP,dim=0)
        recalls = TP_cumsum/ (total_true_bboxes + epsilon)
        precisions = torch.divide(TP_cumsum, (TP_cumsum + FP_cumsum + epsilon))
        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))
        average_precisions.append(torch.trapz(precisions,recalls))

    return  sum(average_precisions) / len(average_precisions)

# https://blog.csdn.net/qq_38669138/article/details/117968834


# https://medium.com/curiosity-and-exploration/mean-average-precision-map-%E8%A9%95%E4%BC%B0%E7%89%A9%E9%AB%94%E5%81%B5%E6%B8%AC%E6%A8%A1%E5%9E%8B%E5%A5%BD%E5%A3%9E%E7%9A%84%E6%8C%87%E6%A8%99-70a2d2872eb0
# https://towardsdatascience.com/breaking-down-mean-average-precision-map-ae462f623a52
# https://zhuanlan.zhihu.com/p/56961620