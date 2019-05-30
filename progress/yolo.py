def loss_function(p, a, obj, lp=1, lxy=1, lhw=0.6, lc=0.4):
    '''
        Send as Numpy Arrays
        p=predicted
        a=actual
        obj=1 if object is actually present
        vectors in order = [p, x, y, h, w, 36 classes] 
        l's are hyperparameters to control the impact of each term
    '''
    
    res = 0
    
    res+=(obj + lc(1-obj)) * (p[0]-a[0])**2
    
    res+=obj*lxy*((p[1]-a[1])**2 + (p[2]-a[2])**2)
    
    res+=obj*lhw*((p[3]-a[3])**2 + (p[4]-a[4])**2)
    
    res+=obj*lp*np.sum((p[5:]-a[5:])**2)
    
    return res



def iou(box1, box2):
    '''
    Implement the intersection over union (IoU) between box1 and box2
    
    box1 = first box, list object with coordinates (x1, y1, x2, y2)
    box2 = second box, list object with coordinates (x1, y1, x2, y2)
    '''

    xi1 = max(box1[0], box2[0])
    yi1 = max(box1[1], box2[1])
    xi2 = min(box1[2], box2[2])
    yi2 = min(box1[3], box2[3])
    inter_area = (xi2 - xi1) * (yi2 - yi1)

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    
    iou = inter_area / union_area
    
    return iou
    