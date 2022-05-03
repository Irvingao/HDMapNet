import torch


def get_batch_iou(pred_map, gt_map):
    '''
    计算交并比
    '''
    intersects = []
    unions = []
    with torch.no_grad():
        pred_map = pred_map.bool()
        gt_map = gt_map.bool()
        # print("pred_map.shape:", pred_map.shape)
        # print("pred_map:", pred_map)
        # print("gt_map.shape:", gt_map.shape)
        # print("gt_map:", gt_map)
        # input()
        
        # 计算一个batch的intersect和union
        for i in range(pred_map.shape[1]): #遍历C (num_classes)遍
            pred = pred_map[:, i] # 获取[B, W, H]
            # print("pred.shape:", pred.shape)
            # print("pred:", pred)
            tgt = gt_map[:, i]
            intersect = (pred & tgt).sum().float()
            union = (pred | tgt).sum().float()
            intersects.append(intersect)
            unions.append(union)

    # print(intersects)
    # print(unions)
    # print(torch.tensor(intersects).shape)
    # print(torch.tensor(unions).shape)
    # input()
    return torch.tensor(intersects), torch.tensor(unions)
