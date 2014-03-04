from scipy import ndimage

def get_segment_masks(seg_labels):
    masks = []
    for label in xrange(seg_labels.max()):
        features, num_features = ndimage.measurements.label(seg_labels == label)
        for i in xrange(1, num_features + 1):
            # to handle the label=0 case
            mask = (features == i)
            size = mask.sum()
            if size > (mask.size / 25):
                masks.append((label + 1) * mask)
    return masks
