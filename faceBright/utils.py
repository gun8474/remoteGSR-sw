# coding: utf-8
import os
import cv2
import numpy as np
import scipy.stats.kde as kde

def calc_min_interval(x, alpha):
    """Internal method to determine the minimum interval of a given width
    Assumes that x is sorted numpy array.
    """

    n = len(x)
    cred_mass = 1.0-alpha

    interval_idx_inc = int(np.floor(cred_mass*n))
    n_intervals = n - interval_idx_inc
    interval_width = x[interval_idx_inc:] - x[:n_intervals]

    if len(interval_width) == 0:
        raise ValueError('Too few elements for interval calculation')

    min_idx = np.argmin(interval_width)
    hdi_min = x[min_idx]
    hdi_max = x[min_idx+interval_idx_inc]
    return hdi_min, hdi_max


def hdi(x, alpha=0.05):
    """Calculate highest posterior density (HPD) of array for given alpha. 
    The HPD is the minimum width Bayesian credible interval (BCI).
    :Arguments:
        x : Numpy array
        An array containing MCMC samples
        alpha : float
        Desired probability of type I error (defaults to 0.05)
    """

    # Make a copy of trace
    x = x.copy()
    # For multivariate node
    if x.ndim > 1:
        # Transpose first, then sort
        tx = np.transpose(x, list(range(x.ndim))[1:]+[0])
        dims = np.shape(tx)
        # Container list for intervals
        intervals = np.resize(0.0, dims[:-1]+(2,))

        for index in make_indices(dims[:-1]):
            try:
                index = tuple(index)
            except TypeError:
                pass

            # Sort trace
            sx = np.sort(tx[index])
            # Append to list
            intervals[index] = calc_min_interval(sx, alpha)
        # Transpose back before returning
        return np.array(intervals)
    else:
        # Sort univariate node
        sx = np.sort(x)
        return np.array(calc_min_interval(sx, alpha))


def hdi2(sample, alpha=0.05, roundto=2):
    """Calculate highest posterior density (HPD) of array for given alpha. 
    The HPD is the minimum width Bayesian credible interval (BCI). 
    The function works for multimodal distributions, returning more than one mode
    Parameters
    ----------
    
    sample : Numpy array or python list
        An array containing MCMC samples
    alpha : float
        Desired probability of type I error (defaults to 0.05)
    roundto: integer
        Number of digits after the decimal point for the results
    Returns
    ----------
    hpd: array with the lower 
          
    """
    sample = np.asarray(sample)
    sample = sample[~np.isnan(sample)]
    # get upper and lower bounds
    l = np.min(sample)
    u = np.max(sample)
    density = kde.gaussian_kde(sample)
    x = np.linspace(l, u, 2000)
    y = density.evaluate(x)
    #y = density.evaluate(x, l, u) waitting for PR to be accepted
    xy_zipped = zip(x, y/np.sum(y))
    xy = sorted(xy_zipped, key=lambda x: x[1], reverse=True)
    xy_cum_sum = 0
    hdv = []

    for val in xy:
        xy_cum_sum += val[1]
        hdv.append(val[0])
        if xy_cum_sum >= (1-alpha):
            break
    hdv.sort()
    diff = (u-l)/20  # differences of 5%
    hpd = []
    hpd.append(round(min(hdv), roundto))

    for i in range(1, len(hdv)):
        if hdv[i]-hdv[i-1] >= diff:
            hpd.append(round(hdv[i-1], roundto))
            hpd.append(round(hdv[i], roundto))
    hpd.append(round(max(hdv), roundto))
    ite = iter(hpd)
    hpd = list(zip(ite, ite))
    modes = []
    for value in hpd:
         x_hpd = x[(x > value[0]) & (x < value[1])]
         y_hpd = y[(x > value[0]) & (x < value[1])]

         modes.append(round(x_hpd[np.argmax(y_hpd)], roundto))
    return hpd, x, y, modes

def mkdir(d):
    os.makedirs(d, exist_ok=True)

def area_of(left_top, right_bottom):
    """Compute the areas of rectangles given two corners.
    Args:
        left_top (N, 2): left top corner.
        right_bottom (N, 2): right bottom corner.
    Returns:
        area (N): return the area.
    """
    hw = np.clip(right_bottom - left_top, 0.0, None)
    return hw[..., 0] * hw[..., 1]
    
def iou_of(boxes0, boxes1, eps=1e-5):
    """Return intersection-over-union (Jaccard index) of boxes.
    Args:
        boxes0 (N, 4): ground truth boxes.
        boxes1 (N or 1, 4): predicted boxes.
        eps: a small number to avoid 0 as denominator.
    Returns:
        iou (N): IoU values.
    """
    overlap_left_top = np.maximum(boxes0[..., :2], boxes1[..., :2])
    overlap_right_bottom = np.minimum(boxes0[..., 2:], boxes1[..., 2:])

    overlap_area = area_of(overlap_left_top, overlap_right_bottom)
    area0 = area_of(boxes0[..., :2], boxes0[..., 2:])
    area1 = area_of(boxes1[..., :2], boxes1[..., 2:])
    return overlap_area / (area0 + area1 - overlap_area + eps)

def hard_nms(box_scores, iou_threshold, top_k=-1, candidate_size=200):
    """
    Args:
        box_scores (N, 5): boxes in corner-form and probabilities.
        iou_threshold: intersection over union threshold.
        top_k: keep top_k results. If k <= 0, keep all the results.
        candidate_size: only consider the candidates with the highest scores.
    Returns:
         picked: a list of indexes of the kept boxes
    """
    scores = box_scores[:, -1]
    boxes = box_scores[:, :-1]
    picked = []
    # _, indexes = scores.sort(descending=True)
    indexes = np.argsort(scores)
    # indexes = indexes[:candidate_size]
    indexes = indexes[-candidate_size:]
    while len(indexes) > 0:
        # current = indexes[0]
        current = indexes[-1]
        picked.append(current)
        if 0 < top_k == len(picked) or len(indexes) == 1:
            break
        current_box = boxes[current, :]
        # indexes = indexes[1:]
        indexes = indexes[:-1]
        rest_boxes = boxes[indexes, :]
        iou = iou_of(
            rest_boxes,
            np.expand_dims(current_box, axis=0),
        )
        indexes = indexes[iou <= iou_threshold]

    return box_scores[picked, :]

def predict(width, height, confidences, boxes, prob_threshold, iou_threshold=0.3, top_k=-1):
    boxes = boxes[0]
    confidences = confidences[0]
    picked_box_probs = []
    picked_labels = []
    for class_index in range(1, confidences.shape[1]):
        probs = confidences[:, class_index]
        mask = probs > prob_threshold
        probs = probs[mask]
        if probs.shape[0] == 0:
            continue
        subset_boxes = boxes[mask, :]
        box_probs = np.concatenate([subset_boxes, probs.reshape(-1, 1)], axis=1)
        box_probs = hard_nms(box_probs,
                                       iou_threshold=iou_threshold,
                                       top_k=top_k,
                                       )
        picked_box_probs.append(box_probs)
        picked_labels.extend([class_index] * box_probs.shape[0])
    if not picked_box_probs:
        return np.array([]), np.array([]), np.array([])
    picked_box_probs = np.concatenate(picked_box_probs)
    picked_box_probs[:, 0] *= width
    picked_box_probs[:, 1] *= height
    picked_box_probs[:, 2] *= width
    picked_box_probs[:, 3] *= height
    return picked_box_probs[:, :4].astype(np.int32), np.array(picked_labels), picked_box_probs[:, 4]


# Plot values in opencv program
class Plotter:
    def __init__(self, plot_width, plot_height, num_plot_values):
        self.width = plot_width
        self.height = plot_height
        self.color_list = [(255, 255, 255), (0, 0, 250), (0, 250, 0), (0, 35, 35), (35, 0, 35), (250, 0, 0),
                           (35, 35, 0)]
        self.color = []
        self.val = []
        self.plot = np.ones((self.height, self.width, 3)) * 0
        self.scale = 10

        for i in range(num_plot_values):
            self.color.append(self.color_list[i])

    def multiplot(self, val, label="plot"):
        self.val.append(val)
        while len(self.val) > self.width:
            self.val.pop(0)

        self.show_plot(label)

    def show_plot(self, label):
        c_width = self.width * self.scale
        self.plot = np.ones((self.height, c_width, 3)) * 0

        self.plot[:, int(c_width * 0.795), :] = 255
        self.plot[:, int(c_width) - 1, :] = 255
        self.plot[0:4, int(c_width * 0.795):, :] = 255
        self.plot[-3:, int(c_width * 0.795):, :] = 255

        for i in range(len(self.val) - 1):
            for j in range(len(self.val[0])):
                cv2.line(self.plot, (i * int(self.scale * 0.8), int(self.height + 2) - int(self.val[i][j] * 3)),
                         ((i + 1) * int(self.scale * 0.8), int(self.height + 2) - int(self.val[i + 1][j] * 3)),
                         self.color[j], 3, cv2.LINE_AA)

        if len(self.val) > 30:
            exp_score = np.mean(self.val[-30:], axis=0)
            cv2.putText(self.plot, 'NEUTRAL: {}'.format(int(exp_score[0])), (int(c_width * 0.8), 35),
                        cv2.FONT_HERSHEY_PLAIN, 1.8, self.color[0], 2, cv2.LINE_AA)
            cv2.putText(self.plot, 'HAPPY: {}'.format(int(exp_score[1])), (int(c_width * 0.8), 70),
                        cv2.FONT_HERSHEY_PLAIN, 1.8, self.color[1], 2, cv2.LINE_AA)
            cv2.putText(self.plot, 'SURPRISE: {}'.format(int(exp_score[2])), (int(c_width * 0.8), 110),
                        cv2.FONT_HERSHEY_PLAIN, 1.8, self.color[2], 2, cv2.LINE_AA)
            cv2.putText(self.plot, 'SADNESS: {}'.format(int(exp_score[3])), (int(c_width * 0.8), 150),
                        cv2.FONT_HERSHEY_PLAIN, 1.8, self.color[3], 2, cv2.LINE_AA)
            cv2.putText(self.plot, 'ANGER: {}'.format(int(exp_score[4])), (int(c_width * 0.8), 200),
                        cv2.FONT_HERSHEY_PLAIN, 1.8, self.color[4], 2, cv2.LINE_AA)
            cv2.putText(self.plot, 'DISGUST: {}'.format(int(exp_score[5])), (int(c_width * 0.8), 240),
                        cv2.FONT_HERSHEY_PLAIN, 1.8, self.color[5], 2, cv2.LINE_AA)
            cv2.putText(self.plot, 'FEAR: {}'.format(int(exp_score[6])), (int(c_width * 0.8), 275),
                        cv2.FONT_HERSHEY_PLAIN, 1.8, self.color[6], 2, cv2.LINE_AA)

        resized = cv2.resize(self.plot, (640, 280), interpolation=cv2.INTER_LINEAR)
        cv2.imshow(label, resized)
