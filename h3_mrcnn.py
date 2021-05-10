#!/usr/bin/env python3.6

import numpy as np
import os
import glob
import cv2
import xmltodict
import zipfile
import shutil
import tensorflow as tf
import numpy
import math
from tqdm import tqdm
from tensorflow.keras import layers as layers
from tensorflow.keras import models as models
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils
import scipy
import skimage.color
import skimage.io
import skimage.transform
import warnings
from distutils.version import LooseVersion


"""
Path and hyperparameter must be global. 
Don't change the location, just edit the value here
"""
#### Question (c): your implementation starts here (don't delete this line)

data_path = './coco/'
ckpt_path = './ckpt/'

epochs = 10
learning_rate = 0.001
batch_size = 2

#### Question (c): your implementation ends here (don't delete this line)


def norm_boxes_graph(boxes, shape):
    """
    Converts boxes from pixel coordinates to normalized coordinates.
    
    Inputs:
    - boxes: [..., (y1, x1, y2, x2)] in pixel coordinates
    - shape: [..., (height, width)] in pixels

    Returns:
    - [..., (y1, x1, y2, x2)] in normalized coordinates
    """
    h, w = tf.split(tf.cast(shape, tf.float32), 2)
    scale = tf.concat([h, w, h, w], axis=-1) - tf.constant(1.0)
    shift = tf.constant([0., 0., 1., 1.])
    return tf.divide(boxes - shift, scale)

def fpn_classifier_graph(rois, feature_maps, image_meta,
                         pool_size, num_classes, train_bn=True,
                         fc_layers_size=1024):
    """
    Builds the computation graph of the feature pyramid network classifier
    and regressor heads.
    
    Inputs:
    - rois: [batch, num_rois, (y1, x1, y2, x2)] Proposal boxes in normalized
          coordinates.
    - feature_maps: List of feature maps from different layers of the pyramid,
                  [P2, P3, P4, P5]. Each has a different resolution.
    - image_meta: [batch, (meta data)] Image details. See compose_image_meta()
    - pool_size: The width of the square feature map generated from ROI Pooling.
    - num_classes: number of classes, which determines the depth of the results
    - train_bn: Boolean. Train or freeze Batch Norm layers
    - fc_layers_size: Size of the 2 FC layers
    
    Returns:
    - logits: [batch, num_rois, NUM_CLASSES] classifier logits (before softmax)
    - probs: [batch, num_rois, NUM_CLASSES] classifier probabilities
    - bbox_deltas: [batch, num_rois, NUM_CLASSES, (dy, dx, log(dh), log(dw))] Deltas to apply to
                     proposal boxes
    """

    x = PyramidROIAlign([pool_size, pool_size])([rois, image_meta] + feature_maps)
    x = layers.TimeDistributed(layers.Conv2D(fc_layers_size, (pool_size, pool_size), padding="valid"),
                           name="mrcnn_class_conv1")(x)
    x = layers.TimeDistributed(BatchNorm(), name='mrcnn_class_bn1')(inputs=x, training=train_bn)
    x = layers.Activation('relu')(x)
    x = layers.TimeDistributed(layers.Conv2D(fc_layers_size, (1, 1)),
                           name="mrcnn_class_conv2")(x)
    x = layers.TimeDistributed(BatchNorm(), name='mrcnn_class_bn2')(inputs=x, training=train_bn)
    x = layers.Activation('relu')(x)

    shared = layers.Lambda(lambda x: tf.keras.backend.squeeze(tf.keras.backend.squeeze(x, 3), 2),
                       name="pool_squeeze")(x)

    mrcnn_class_logits = layers.TimeDistributed(layers.Dense(num_classes),
                                            name='mrcnn_class_logits')(shared)
    mrcnn_probs = layers.TimeDistributed(layers.Activation("softmax"),
                                     name="mrcnn_class")(mrcnn_class_logits)


    x = layers.TimeDistributed(layers.Dense(num_classes * 4, activation='linear'),
                           name='mrcnn_bbox_fc')(shared)

    s = tf.keras.backend.int_shape(x)
    if s[1]==None:
        mrcnn_bbox = layers.Reshape((-1, num_classes, 4), name="mrcnn_bbox")(x)
    else:
        mrcnn_bbox = layers.Reshape((s[1], num_classes, 4), name="mrcnn_bbox")(x)

    return mrcnn_class_logits, mrcnn_probs, mrcnn_bbox

def box_refinement_graph(box, gt_box):
    """
    Compute refinement needed to transform box to gt_box.
    
    Inputs:
    -box: [N, (y1, x1, y2, x2)]
    -gt_box: [N, (y1, x1, y2, x2)]
    
    Returns:
    -result
    """
    box = tf.cast(box, tf.float32)
    gt_box = tf.cast(gt_box, tf.float32)

    height = box[:, 2] - box[:, 0]
    width = box[:, 3] - box[:, 1]
    center_y = box[:, 0] + 0.5 * height
    center_x = box[:, 1] + 0.5 * width

    gt_height = gt_box[:, 2] - gt_box[:, 0]
    gt_width = gt_box[:, 3] - gt_box[:, 1]
    gt_center_y = gt_box[:, 0] + 0.5 * gt_height
    gt_center_x = gt_box[:, 1] + 0.5 * gt_width

    dy = (gt_center_y - center_y) / height
    dx = (gt_center_x - center_x) / width
    dh = tf.math.log(gt_height / height)
    dw = tf.math.log(gt_width / width)

    result = tf.stack([dy, dx, dh, dw], axis=1)
    return result

def trim_zeros_graph(boxes, name='trim_zeros'):
    """
    Often boxes are represented with matrices of shape [N, 4] and
    are padded with zeros. This removes zero boxes.
    
    Inputs:
    - boxes: [N, 4] matrix of boxes.
    
    Returns:
    - boxes: [N, 4] matrix of boxes.
    - non_zeros: [N] a 1D boolean mask identifying the rows to keep
    """
    non_zeros = tf.cast(tf.reduce_sum(tf.abs(boxes), axis=1), tf.bool)
    boxes = tf.boolean_mask(boxes, non_zeros, name=name)
    return boxes, non_zeros

def overlaps_graph(boxes1, boxes2):
    """
    Computes IoU overlaps between two sets of boxes.
    
    Inputs:
    - boxes1, boxes2: [N, (y1, x1, y2, x2)].
    
    Returns:
    - overlaps
    """

    b1 = tf.reshape(tf.tile(tf.expand_dims(boxes1, 1),
                            [1, 1, tf.shape(boxes2)[0]]), [-1, 4])
    b2 = tf.tile(boxes2, [tf.shape(boxes1)[0], 1])

    b1_y1, b1_x1, b1_y2, b1_x2 = tf.split(b1, 4, axis=1)
    b2_y1, b2_x1, b2_y2, b2_x2 = tf.split(b2, 4, axis=1)
    y1 = tf.maximum(b1_y1, b2_y1)
    x1 = tf.maximum(b1_x1, b2_x1)
    y2 = tf.minimum(b1_y2, b2_y2)
    x2 = tf.minimum(b1_x2, b2_x2)
    intersection = tf.maximum(x2 - x1, 0) * tf.maximum(y2 - y1, 0)

    b1_area = (b1_y2 - b1_y1) * (b1_x2 - b1_x1)
    b2_area = (b2_y2 - b2_y1) * (b2_x2 - b2_x1)
    union = b1_area + b2_area - intersection

    iou = intersection / union
    overlaps = tf.reshape(iou, [tf.shape(boxes1)[0], tf.shape(boxes2)[0]])
    return overlaps


def detection_targets_graph(proposals, gt_class_ids, gt_boxes, gt_masks):
    """
    Generates detection targets for one image. Subsamples proposals and
    generates target class IDs, bounding box deltas, and masks for each.
    
    Inputs:
    - proposals: [POST_NMS_ROIS_TRAINING, (y1, x1, y2, x2)] in normalized coordinates. Might
               be zero padded if there are not enough proposals.
    - gt_class_ids: [MAX_GT_INSTANCES] int class IDs
    - gt_boxes: [MAX_GT_INSTANCES, (y1, x1, y2, x2)] in normalized coordinates.
    - gt_masks: [height, width, MAX_GT_INSTANCES] of boolean type.
    
    Returns: Target ROIs and corresponding class IDs, bounding box shifts, and masks.
    - rois: [TRAIN_ROIS_PER_IMAGE, (y1, x1, y2, x2)] in normalized coordinates
    - class_ids: [TRAIN_ROIS_PER_IMAGE]. Integer class IDs. Zero padded.
    - deltas: [TRAIN_ROIS_PER_IMAGE, (dy, dx, log(dh), log(dw))]
    - masks: [TRAIN_ROIS_PER_IMAGE, height, width]. Masks cropped to bbox
           boundaries and resized to neural network output size.
    """

    asserts = [
        tf.Assert(tf.greater(tf.shape(proposals)[0], 0), [proposals],
                  name="roi_assertion"),
    ]
    with tf.control_dependencies(asserts):
        proposals = tf.identity(proposals)


    proposals, _ = trim_zeros_graph(proposals, name="trim_proposals")
    gt_boxes, non_zeros = trim_zeros_graph(gt_boxes, name="trim_gt_boxes")
    gt_class_ids = tf.boolean_mask(gt_class_ids, non_zeros,
                                   name="trim_gt_class_ids")
    gt_masks = tf.gather(gt_masks, tf.where(non_zeros)[:, 0], axis=2,
                         name="trim_gt_masks")

    crowd_ix = tf.where(gt_class_ids < 0)[:, 0]
    non_crowd_ix = tf.where(gt_class_ids > 0)[:, 0]
    crowd_boxes = tf.gather(gt_boxes, crowd_ix)
    gt_class_ids = tf.gather(gt_class_ids, non_crowd_ix)
    gt_boxes = tf.gather(gt_boxes, non_crowd_ix)
    gt_masks = tf.gather(gt_masks, non_crowd_ix, axis=2)

    overlaps = overlaps_graph(proposals, gt_boxes)

    crowd_overlaps = overlaps_graph(proposals, crowd_boxes)
    crowd_iou_max = tf.reduce_max(crowd_overlaps, axis=1)
    no_crowd_bool = (crowd_iou_max < 0.001)

    roi_iou_max = tf.reduce_max(overlaps, axis=1)

    positive_roi_bool = (roi_iou_max >= 0.5)
    positive_indices = tf.where(positive_roi_bool)[:, 0]

    negative_indices = tf.where(tf.logical_and(roi_iou_max < 0.5, no_crowd_bool))[:, 0]


    positive_count = int(200 *
                         0.33)
    positive_indices = tf.random.shuffle(positive_indices)[:positive_count]
    positive_count = tf.shape(positive_indices)[0]

    r = 1.0 / 0.33
    negative_count = tf.cast(r * tf.cast(positive_count, tf.float32), tf.int32) - positive_count
    negative_indices = tf.random.shuffle(negative_indices)[:negative_count]

    positive_rois = tf.gather(proposals, positive_indices)
    negative_rois = tf.gather(proposals, negative_indices)

    positive_overlaps = tf.gather(overlaps, positive_indices)
    roi_gt_box_assignment = tf.cond(
        tf.greater(tf.shape(positive_overlaps)[1], 0),
        true_fn = lambda: tf.argmax(positive_overlaps, axis=1),
        false_fn = lambda: tf.cast(tf.constant([]),tf.int64)
    )
    roi_gt_boxes = tf.gather(gt_boxes, roi_gt_box_assignment)
    roi_gt_class_ids = tf.gather(gt_class_ids, roi_gt_box_assignment)


    deltas = box_refinement_graph(positive_rois, roi_gt_boxes)
    deltas /= np.array([0.1, 0.1, 0.2, 0.2])


    transposed_masks = tf.expand_dims(tf.transpose(gt_masks, [2, 0, 1]), -1)
    roi_masks = tf.gather(transposed_masks, roi_gt_box_assignment)

    boxes = positive_rois
    box_ids = tf.range(0, tf.shape(roi_masks)[0])
    masks = tf.image.crop_and_resize(tf.cast(roi_masks, tf.float32), boxes,
                                     box_ids,
                                     [28,28])
    masks = tf.squeeze(masks, axis=3)

    masks = tf.round(masks)

    rois = tf.concat([positive_rois, negative_rois], axis=0)
    N = tf.shape(negative_rois)[0]
    P = tf.maximum(200 - tf.shape(rois)[0], 0)
    rois = tf.pad(rois, [(0, P), (0, 0)])
    roi_gt_boxes = tf.pad(roi_gt_boxes, [(0, N + P), (0, 0)])
    roi_gt_class_ids = tf.pad(roi_gt_class_ids, [(0, N + P)])
    deltas = tf.pad(deltas, [(0, N + P), (0, 0)])
    masks = tf.pad(masks, [[0, N + P], (0, 0), (0, 0)])

    return rois, roi_gt_class_ids, deltas, masks

def batch_slice(inputs, graph_fn, batch_size, names=None):
    """
    Splits inputs into slices and feeds each slice to a copy of the given
    computation graph and then combines the results. It allows you to run a
    graph on a batch of inputs even if the graph is written to support one
    instance only.
    
    Inputs:
    - inputs: list of tensors. All must have the same first dimension length
    - graph_fn: A function that returns a TF tensor that's part of a graph.
    - batch_size: number of slices to divide the data into.
    - names: If provided, assigns names to the resulting tensors.
    
    Returns:
    - result
    """
    if not isinstance(inputs, list):
        inputs = [inputs]

    outputs = []
    for i in range(batch_size):
        inputs_slice = [x[i] for x in inputs]
        output_slice = graph_fn(*inputs_slice)
        if not isinstance(output_slice, (tuple, list)):
            output_slice = [output_slice]
        outputs.append(output_slice)
    outputs = list(zip(*outputs))

    if names is None:
        names = [None] * len(outputs)

    result = [tf.stack(o, axis=0, name=n)
              for o, n in zip(outputs, names)]
    if len(result) == 1:
        result = result[0]

    return result

def build_fpn_mask_graph(rois, feature_maps, image_meta,
                         pool_size, num_classes, train_bn=True):
    """
    Builds the computation graph of the mask head of Feature Pyramid Network.
    
    Inputs:
    - rois: [batch, num_rois, (y1, x1, y2, x2)] Proposal boxes in normalized
          coordinates.
    - feature_maps: List of feature maps from different layers of the pyramid,
                  [P2, P3, P4, P5]. Each has a different resolution.
    - image_meta: [batch, (meta data)] Image details. See compose_image_meta()
    - pool_size: The width of the square feature map generated from ROI Pooling.
    - num_classes: number of classes, which determines the depth of the results
    - train_bn: Boolean. Train or freeze Batch Norm layers
    
    Returns: 
    - x: Masks of [batch, num_rois, MASK_POOL_SIZE, MASK_POOL_SIZE, NUM_CLASSES]
    """

    x = PyramidROIAlign([pool_size, pool_size])([rois, image_meta] + feature_maps)

    x = layers.TimeDistributed(layers.Conv2D(256, (3, 3), padding="same"))(x)
    x = layers.TimeDistributed(BatchNorm())(inputs=x, training=train_bn)
    x = layers.Activation('relu')(x)

    x = layers.TimeDistributed(layers.Conv2D(256, (3, 3), padding="same"))(x)
    x = layers.TimeDistributed(BatchNorm())(inputs=x, training=train_bn)
    x = layers.Activation('relu')(x)

    x = layers.TimeDistributed(layers.Conv2D(256, (3, 3), padding="same"))(x)
    x = layers.TimeDistributed(BatchNorm())(inputs=x, training=train_bn)
    x = layers.Activation('relu')(x)

    x = layers.TimeDistributed(layers.Conv2D(256, (3, 3), padding="same"))(x)
    x = layers.TimeDistributed(BatchNorm())(inputs=x, training=train_bn)
    x = layers.Activation('relu')(x)

    x = layers.TimeDistributed(layers.Conv2DTranspose(256, (2, 2), strides=2, activation="relu"))(x)
    x = layers.TimeDistributed(layers.Conv2D(num_classes, (1, 1), strides=1, activation="sigmoid"))(x)
    return x

def batch_pack_graph(x, counts, num_rows):
    """
    Picks different number of values from each row
    in x depending on the values in counts.
    """
    outputs = []
    for i in range(num_rows):
        outputs.append(x[i, :counts[i]])
    return tf.concat(outputs, axis=0)

def smooth_l1_loss(y_true, y_pred):
    """
    Implements Smooth-L1 loss.
    y_true and y_pred are typically: [N, 4], but could be any shape.
    """
    diff = tf.keras.backend.abs(y_true - y_pred)
    less_than_one = tf.keras.backend.cast(tf.keras.backend.less(diff, 1.0), "float32")
    loss = (less_than_one * 0.5 * diff**2) + (1 - less_than_one) * (diff - 0.5)
    return loss


def rpn_class_loss_graph(rpn_match, rpn_class_logits):
    """
    RPN anchor classifier loss.
    
    Inputs:
    - rpn_match: [batch, anchors, 1]. Anchor match type. 1=positive,
               -1=negative, 0=neutral anchor.
    - rpn_class_logits: [batch, anchors, 2]. RPN classifier logits for BG/FG.
    
    Retruns:
    - loss    
    """

    rpn_match = tf.squeeze(rpn_match, -1)

    anchor_class = tf.keras.backend.cast(tf.keras.backend.equal(rpn_match, 1), tf.int32)

    indices = tf.where(tf.keras.backend.not_equal(rpn_match, 0))

    rpn_class_logits = tf.gather_nd(rpn_class_logits, indices)
    anchor_class = tf.gather_nd(anchor_class, indices)

    loss = tf.keras.backend.sparse_categorical_crossentropy(target=anchor_class,
                                             output=rpn_class_logits,
                                             from_logits=True)
    loss = tf.keras.backend.switch(tf.size(loss) > 0, tf.keras.backend.mean(loss), tf.constant(0.0))
    return loss


def rpn_bbox_loss_graph(target_bbox, rpn_match, rpn_bbox):
    """
    Return the RPN bounding box loss graph.
    
    Inputs:
    - target_bbox: [batch, max positive anchors, (dy, dx, log(dh), log(dw))].
        Uses 0 padding to fill in unsed bbox deltas.
    - rpn_match: [batch, anchors, 1]. Anchor match type. 1=positive,
               -1=negative, 0=neutral anchor.
    - rpn_bbox: [batch, anchors, (dy, dx, log(dh), log(dw))]
    
    Returns:
    - loss
    """

    rpn_match = tf.keras.backend.squeeze(rpn_match, -1)
    indices = tf.where(tf.keras.backend.equal(rpn_match, 1))

    rpn_bbox = tf.gather_nd(rpn_bbox, indices)

    batch_counts = tf.keras.backend.sum(tf.keras.backend.cast(tf.keras.backend.equal(rpn_match, 1), tf.int32), axis=1)
    target_bbox = batch_pack_graph(target_bbox, batch_counts, batch_size)

    loss = smooth_l1_loss(target_bbox, rpn_bbox)
    
    loss = tf.keras.backend.switch(tf.size(loss) > 0, tf.keras.backend.mean(loss), tf.constant(0.0))
    return loss


def mrcnn_class_loss_graph(target_class_ids, pred_class_logits,
                           active_class_ids):
    """
    Loss for the classifier head of Mask RCNN.
    
    Inputs:
    - target_class_ids: [batch, num_rois]. Integer class IDs. Uses zero
        padding to fill in the array.
    - pred_class_logits: [batch, num_rois, num_classes]
    - active_class_ids: [batch, num_classes]. Has a value of 1 for
        classes that are in the dataset of the image, and 0
        for classes that are not in the dataset.
        
    Returns:
    - loss
    """

    target_class_ids = tf.cast(target_class_ids, 'int64')

    pred_class_ids = tf.argmax(pred_class_logits, axis=2)

    pred_active = tf.gather(active_class_ids[0], pred_class_ids)


    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=target_class_ids, logits=pred_class_logits)


    loss = loss * pred_active


    loss = tf.reduce_sum(loss) / tf.reduce_sum(pred_active)
    return loss


def mrcnn_bbox_loss_graph(target_bbox, target_class_ids, pred_bbox):
    """
    Loss for Mask R-CNN bounding box refinement.
    
    Inputs:
    - target_bbox: [batch, num_rois, (dy, dx, log(dh), log(dw))]
    - target_class_ids: [batch, num_rois]. Integer class IDs.
    - pred_bbox: [batch, num_rois, num_classes, (dy, dx, log(dh), log(dw))]
    
    Returns:
    - loss
    """
    
    target_class_ids = tf.keras.backend.reshape(target_class_ids, (-1,))
    target_bbox = tf.keras.backend.reshape(target_bbox, (-1, 4))
    pred_bbox = tf.keras.backend.reshape(pred_bbox, (-1, tf.keras.backend.int_shape(pred_bbox)[2], 4))

    
    positive_roi_ix = tf.where(target_class_ids > 0)[:, 0]
    positive_roi_class_ids = tf.cast(
        tf.gather(target_class_ids, positive_roi_ix), tf.int64)
    indices = tf.stack([positive_roi_ix, positive_roi_class_ids], axis=1)

    
    target_bbox = tf.gather(target_bbox, positive_roi_ix)
    pred_bbox = tf.gather_nd(pred_bbox, indices)

    
    loss = tf.keras.backend.switch(tf.size(target_bbox) > 0,
                    smooth_l1_loss(y_true=target_bbox, y_pred=pred_bbox),
                    tf.constant(0.0))
    loss = tf.keras.backend.mean(loss)
    return loss


def mrcnn_mask_loss_graph(target_masks, target_class_ids, pred_masks):
    """
    Mask binary cross-entropy loss for the masks head.
    
    Inputs:
    - target_masks: [batch, num_rois, height, width].
        A float32 tensor of values 0 or 1. Uses zero padding to fill array.
    - target_class_ids: [batch, num_rois]. Integer class IDs. Zero padded.
    - pred_masks: [batch, proposals, height, width, num_classes] float32 tensor
                with values from 0 to 1.
                
    Returns:
    - loss
    """
    
    target_class_ids = tf.keras.backend.reshape(target_class_ids, (-1,))
    mask_shape = tf.shape(target_masks)
    target_masks = tf.keras.backend.reshape(target_masks, (-1, mask_shape[2], mask_shape[3]))
    pred_shape = tf.shape(pred_masks)
    pred_masks = tf.keras.backend.reshape(pred_masks,
                           (-1, pred_shape[2], pred_shape[3], pred_shape[4]))
    
    pred_masks = tf.transpose(pred_masks, [0, 3, 1, 2])


    positive_ix = tf.where(target_class_ids > 0)[:, 0]
    positive_class_ids = tf.cast(
        tf.gather(target_class_ids, positive_ix), tf.int64)
    indices = tf.stack([positive_ix, positive_class_ids], axis=1)

   
    y_true = tf.gather(target_masks, positive_ix)
    y_pred = tf.gather_nd(pred_masks, indices)

    
    loss = tf.keras.backend.switch(tf.size(y_true) > 0,
                    tf.keras.backend.binary_crossentropy(target=y_true, output=y_pred),
                    tf.constant(0.0))
    loss = tf.keras.backend.mean(loss)
    return loss

def apply_box_deltas_graph(boxes, deltas):
    """
    Applies the given deltas to the given boxes.
    
    Inputs:
    - boxes: [N, (y1, x1, y2, x2)] boxes to update
    - deltas: [N, (dy, dx, log(dh), log(dw))] refinements to apply
    
    Returns:
    - result
    """
    
    height = boxes[:, 2] - boxes[:, 0]
    width = boxes[:, 3] - boxes[:, 1]
    center_y = boxes[:, 0] + 0.5 * height
    center_x = boxes[:, 1] + 0.5 * width

    center_y += deltas[:, 0] * height
    center_x += deltas[:, 1] * width
    height *= tf.exp(deltas[:, 2])
    width *= tf.exp(deltas[:, 3])

    y1 = center_y - 0.5 * height
    x1 = center_x - 0.5 * width
    y2 = y1 + height
    x2 = x1 + width
    result = tf.stack([y1, x1, y2, x2], axis=1, name="apply_box_deltas_out")
    return result


def clip_boxes_graph(boxes, window):
    """
    Inputs:
    - boxes: [N, (y1, x1, y2, x2)]
    - window: [4] in the form y1, x1, y2, x2
    """

    wy1, wx1, wy2, wx2 = tf.split(window, 4)
    y1, x1, y2, x2 = tf.split(boxes, 4, axis=1)

    y1 = tf.maximum(tf.minimum(y1, wy2), wy1)
    x1 = tf.maximum(tf.minimum(x1, wx2), wx1)
    y2 = tf.maximum(tf.minimum(y2, wy2), wy1)
    x2 = tf.maximum(tf.minimum(x2, wx2), wx1)
    clipped = tf.concat([y1, x1, y2, x2], axis=1, name="clipped_boxes")
    clipped.set_shape((clipped.shape[0], 4))
    return clipped

def norm_boxes(boxes, shape):
    """
    Converts boxes from pixel coordinates to normalized coordinates.
    
    Inputs:
    - boxes: [N, (y1, x1, y2, x2)] in pixel coordinates
    - shape: [..., (height, width)] in pixels
    
    Returns:
    - [N, (y1, x1, y2, x2)] in normalized coordinates
    """
    h, w = shape
    scale = np.array([h - 1, w - 1, h - 1, w - 1])
    shift = np.array([0, 0, 1, 1])
    return np.divide((boxes - shift), scale).astype(np.float32)

def parse_image_meta_graph(meta):
    """
    Parses a tensor that contains image attributes to its components.
    See compose_image_meta() for more details.
    
    Inputs:
    - meta: [batch, meta length] where meta length depends on NUM_CLASSES
    
    Returns:
    - a dict of the parsed tensors.
    """
    image_id = meta[:, 0]
    original_image_shape = meta[:, 1:4]
    image_shape = meta[:, 4:7]
    window = meta[:, 7:11]  
    scale = meta[:, 11]
    active_class_ids = meta[:, 12:]
    return {
        "image_id": image_id,
        "original_image_shape": original_image_shape,
        "image_shape": image_shape,
        "window": window,
        "scale": scale,
        "active_class_ids": active_class_ids,
    }

def identity_block(input_tensor, kernel_size, filters, stage, block,
                   use_bias=True, train_bn=True):
    """
    The identity_block is the block that has no conv layer at shortcut
    
    Inputs:
    - input_tensor: input tensor
    - kernel_size: default 3, the kernel size of middle conv layer at main path
    - filters: list of integers, the nb_filters of 3 conv layer at main path
    - stage: integer, current stage label, used for generating layer names
    - block: 'a','b'..., current block label, used for generating layer names
    - use_bias: Boolean. To use or not use a bias in conv layers.
    - train_bn: Boolean. Train or freeze Batch Norm layers
    
    Returns:
    - x: output layer
    """
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = layers.Conv2D(nb_filter1, (1, 1), name=conv_name_base + '2a',
                  use_bias=use_bias)(input_tensor)
    x = BatchNorm()(inputs=x, training=train_bn)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same',
                  name=conv_name_base + '2b', use_bias=use_bias)(x)
    x = BatchNorm()(inputs=x, training=train_bn)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c',
                  use_bias=use_bias)(x)
    x = BatchNorm()(inputs=x, training=train_bn)

    x = layers.Add()([x, input_tensor])
    x = layers.Activation('relu', name='res' + str(stage) + block + '_out')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block,
               strides=(2, 2), use_bias=True, train_bn=True):
    """
    conv_block is the block that has a conv layer at shortcut
    
    Inputs:
    - input_tensor: input tensor
    - kernel_size: default 3, the kernel size of middle conv layer at main path
    - filters: list of integers, the nb_filters of 3 conv layer at main path
    - stage: integer, current stage label, used for generating layer names
    - block: 'a','b'..., current block label, used for generating layer names
    - strides
    - use_bias: Boolean. To use or not use a bias in conv layers.
    - train_bn: Boolean. Train or freeze Batch Norm layers

    Returns:
    - x: output layer
    """
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = layers.Conv2D(nb_filter1, (1, 1), strides=strides,
                  name=conv_name_base + '2a', use_bias=use_bias)(input_tensor)
    x = BatchNorm()(inputs=x, training=train_bn)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same',
                  name=conv_name_base + '2b', use_bias=use_bias)(x)
    x = BatchNorm()(inputs=x, training=train_bn)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(nb_filter3, (1, 1), name=conv_name_base +
                  '2c', use_bias=use_bias)(x)
    x = BatchNorm()(inputs=x, training=train_bn)

    shortcut = layers.Conv2D(nb_filter3, (1, 1), strides=strides,
                         name=conv_name_base + '1', use_bias=use_bias)(input_tensor)
    shortcut = BatchNorm()(inputs=shortcut, training=train_bn)

    x = layers.Add()([x, shortcut])
    x = layers.Activation('relu', name='res' + str(stage) + block + '_out')(x)
    return x


def resnet(input_image, train_bn):
    """
    Build a ResNet graph of resnet101.
    The graph consists of 5 stages.
    
    Inputs:
    - Input_image: shape is (batch_size, 1024, 1024, 3)
    - train_bn: Boolean. Train or freeze Batch Norm layers
    
    Outputs:
    - [C1, C2, C3, C4, C5]: each feature map of stage.
        stage 1 = 64 filters of 7*7 conv with stride 2 and same padding, and 3*3 max pooling with stride 2
        stage 2 = 1 conv blocks and 2 identity blocks which outputs C2 with 256 channel size
        stage 3 = 1 conv blocks and 3 identity blocks which outputs C3 with 512 channel size
        stage 4 = 1 conv blocks and 22 identity blocks which outputs C4 with 1024 channel size
        stage 5 = 1 conv blocks and 2 identity blocks which outputs C5 with 2048 channel size

    """

    #### Question (a): your implementation starts here (don't delete this line)

    def _make_layers(input_tensor, n_channels, n_blocks, stage, train_bn, strides=None):
        filters = (n_channels[0], n_channels[0], n_channels[1])
        if strides is not None:
            out = conv_block(input_tensor, 3, filters, stage, "conv", train_bn=train_bn, strides=strides)
        else:
            out = conv_block(input_tensor, 3, filters, stage, "conv", train_bn=train_bn)
        for i in range(n_blocks):
            out = identity_block(out, 3, filters, stage, f"identity{i}", train_bn=train_bn)
        return out

    C1 = layers.Conv2D(64, (7, 7), strides=(2, 2), padding="same")(input_image)
    C1 = BatchNorm()(inputs=C1, training=train_bn)
    C1 = layers.Activation('relu')(C1)
    C1 = layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding="same")(C1)

    C2 = _make_layers(C1, n_channels=(64, 256), n_blocks=2, stage=2, strides=(1, 1), train_bn=train_bn)
    C3 = _make_layers(C2, n_channels=(128, 512), n_blocks=3, stage=3, train_bn=train_bn)
    C4 = _make_layers(C3, n_channels=(256, 1024), n_blocks=22, stage=4, train_bn=train_bn)
    C5 = _make_layers(C4, n_channels=(512, 2048), n_blocks=2, stage=5, train_bn=train_bn)

    #### Question (a): your implementation ends here (don't delete this line)
    
    return [C1, C2, C3, C4, C5]

def rpn_graph(feature_map, anchors_per_location, anchor_stride):
    """
    Builds the computation graph of Region Proposal Network.
    
    Inputs:
    - feature_map: backbone features [batch, height, width, depth]
    - anchors_per_location: number of anchors per pixel in the feature map
    - anchor_stride: Controls the density of anchors. Typically 1 (anchors for
                   every pixel in the feature map), or 2 (every other pixel).
    Returns:
    - rpn_class_logits: [batch, H * W * anchors_per_location, 2] Anchor classifier logits (before softmax)
    - rpn_probs: [batch, H * W * anchors_per_location, 2] Anchor classifier probabilities.
    - rpn_bbox: [batch, H * W * anchors_per_location, (dy, dx, log(dh), log(dw))] Deltas to be
                  applied to anchors.
    """

    shared = layers.Conv2D(512, (3, 3), padding='same', activation='relu',
                       strides=anchor_stride,
                       name='rpn_conv_shared')(feature_map)


    x = layers.Conv2D(2 * anchors_per_location, (1, 1), padding='valid',
                  activation='linear', name='rpn_class_raw')(shared)


    rpn_class_logits = layers.Lambda(
        lambda t: tf.reshape(t, [tf.shape(t)[0], -1, 2]))(x)


    rpn_probs = layers.Activation(
        "softmax", name="rpn_class_xxx")(rpn_class_logits)


    x = layers.Conv2D(anchors_per_location * 4, (1, 1), padding="valid",
                  activation='linear', name='rpn_bbox_pred')(shared)

    rpn_bbox = layers.Lambda(lambda t: tf.reshape(t, [tf.shape(t)[0], -1, 4]))(x)

    return [rpn_class_logits, rpn_probs, rpn_bbox]

def build_rpn_model(anchor_stride, anchors_per_location, depth):
    """
    Builds a Keras model of the Region Proposal Network.
    It wraps the RPN graph so it can be used multiple times with shared
    weights.
    
    Inputs:
    - anchors_per_location: number of anchors per pixel in the feature map
    - anchor_stride: Controls the density of anchors. Typically 1 (anchors for
                   every pixel in the feature map), or 2 (every other pixel).
    - depth: Depth of the backbone feature map.
    
    Returns:
    - Keras Model object. The model outputs, when called, are:
    - rpn_class_logits: [batch, H * W * anchors_per_location, 2] Anchor classifier logits (before softmax)
    - rpn_probs: [batch, H * W * anchors_per_location, 2] Anchor classifier probabilities.
    - rpn_bbox: [batch, H * W * anchors_per_location, (dy, dx, log(dh), log(dw))] Deltas to be
                applied to anchors.
    """
    input_feature_map = layers.Input(shape=[None, None, depth],
                                 name="input_rpn_feature_map")
    outputs = rpn_graph(input_feature_map, anchors_per_location, anchor_stride)
    return tf.keras.models.Model([input_feature_map], outputs, name="rpn_model")


class AnchorsLayer(tf.keras.layers.Layer):
    def __init__(self, name="anchors", **kwargs):
        super(AnchorsLayer, self).__init__(name=name, **kwargs)

    def call(self, anchor):
        return anchor


class BatchNorm(layers.BatchNormalization):
    """
    Extends the Keras BatchNormalization class to allow a central place 
    to make changes if needed.
    Batch normalization has a negative effect on training if batches are small
    so this layer is often frozen (via setting in Config class) and functions
    as linear layer.
    """
    def call(self, inputs, training=None):
      
        return super(self.__class__, self).call(inputs, training=training)


class PyramidROIAlign(layers.Layer):
    """
    Implements ROI Pooling on multiple levels of the feature pyramid.
    
    Inputs:
    - pool_shape: [pool_height, pool_width] of the output pooled regions. Usually [7, 7]
    Inputs:
    - boxes: [batch, num_boxes, (y1, x1, y2, x2)] in normalized
             coordinates. Possibly padded with zeros if not enough
             boxes to fill the array.
    - image_meta: [batch, (meta data)] Image details. See compose_image_meta()
    - feature_maps: List of feature maps from different levels of the pyramid.
                    Each is [batch, height, width, channels]
                    
    Returns:
    - Pooled regions in the shape: [batch, num_boxes, pool_height, pool_width, channels].
    The width and height are those specific in the pool_shape in the layer
    constructor.
    """

    def __init__(self, pool_shape):
        super(PyramidROIAlign, self).__init__()
        self.pool_shape = tuple(pool_shape)

    def call(self, inputs):

        boxes = inputs[0]


        image_meta = inputs[1]


        feature_maps = inputs[2:]


        y1, x1, y2, x2 = tf.split(boxes, 4, axis=2)
        h = y2 - y1
        w = x2 - x1

        image_shape = parse_image_meta_graph(image_meta)['image_shape'][0]

        image_area = tf.cast(image_shape[0] * image_shape[1], tf.float32)
        roi_level = tf.math.log(tf.sqrt(h * w) / tf.math.log(2.0) / (224.0 / tf.sqrt(image_area)))
        roi_level = tf.minimum(5, tf.maximum(
            2, 4 + tf.cast(tf.round(roi_level), tf.int32)))
        roi_level = tf.squeeze(roi_level, 2)

        pooled = []
        box_to_level = []
        for i, level in enumerate(range(2, 6)):
            ix = tf.where(tf.equal(roi_level, level))
            level_boxes = tf.gather_nd(boxes, ix)

            box_indices = tf.cast(ix[:, 0], tf.int32)

            box_to_level.append(ix)

            level_boxes = tf.stop_gradient(level_boxes)
            box_indices = tf.stop_gradient(box_indices)

            pooled.append(tf.image.crop_and_resize(
                feature_maps[i], level_boxes, box_indices, self.pool_shape,
                method="bilinear"))

        pooled = tf.concat(pooled, axis=0)


        box_to_level = tf.concat(box_to_level, axis=0)
        box_range = tf.expand_dims(tf.range(tf.shape(box_to_level)[0]), 1)
        box_to_level = tf.concat([tf.cast(box_to_level, tf.int32), box_range],
                                 axis=1)


        sorting_tensor = box_to_level[:, 0] * 100000 + box_to_level[:, 1]
        ix = tf.nn.top_k(sorting_tensor, k=tf.shape(
            box_to_level)[0]).indices[::-1]
        ix = tf.gather(box_to_level[:, 2], ix)
        pooled = tf.gather(pooled, ix)

        shape = tf.concat([tf.shape(boxes)[:2], tf.shape(pooled)[1:]], axis=0)
        pooled = tf.reshape(pooled, shape)
        return pooled

    def compute_output_shape(self, input_shape):
        return input_shape[0][:2] + self.pool_shape + (input_shape[2][-1], )


class DetectionTargetLayer(layers.Layer):
    """
    Subsamples proposals and generates target box refinement, class_ids,
    and masks for each.
    
    Inputs:
    - proposals: [batch, N, (y1, x1, y2, x2)] in normalized coordinates. Might
               be zero padded if there are not enough proposals.
    - gt_class_ids: [batch, MAX_GT_INSTANCES] Integer class IDs.
    - gt_boxes: [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)] in normalized
              coordinates.
    - gt_masks: [batch, height, width, MAX_GT_INSTANCES] of boolean type
    
    Returns: 
    - Target ROIs and corresponding class IDs, bounding box shifts,
    and masks.
    - rois: [batch, TRAIN_ROIS_PER_IMAGE, (y1, x1, y2, x2)] in normalized
          coordinates
    - target_class_ids: [batch, TRAIN_ROIS_PER_IMAGE]. Integer class IDs.
    - target_deltas: [batch, TRAIN_ROIS_PER_IMAGE, (dy, dx, log(dh), log(dw)]
    - target_mask: [batch, TRAIN_ROIS_PER_IMAGE, height, width]
                 Masks cropped to bbox boundaries and resized to neural
                 network output size.
    """

    def __init__(self):
        super(DetectionTargetLayer, self).__init__()

    def call(self, inputs):
        proposals = inputs[0]
        gt_class_ids = inputs[1]
        gt_boxes = inputs[2]
        gt_masks = inputs[3]

        names = ["rois", "target_class_ids", "target_bbox", "target_mask"]
        outputs = batch_slice(
            [proposals, gt_class_ids, gt_boxes, gt_masks],
            lambda w, x, y, z: detection_targets_graph(
                w, x, y, z), batch_size)
        return outputs

    def compute_output_shape(self, input_shape):
        return [
            (None, 200, 4),  # rois
            (None, 200),  # class_ids
            (None, 200, 4),  # deltas
            (None, 200, 28, 28)  # masks
        ]

    def compute_mask(self, inputs, mask=None):
        return [None, None, None, None]
    
class ProposalLayer(tf.keras.layers.Layer):
    """
    Receives anchor scores and selects a subset to pass as proposals
    to the second stage. Filtering is done based on anchor scores and
    non-max suppression to remove overlaps. It also applies bounding
    box refinement deltas to anchors.
    
    Inputs:
    - rpn_probs: [batch, num_anchors, (bg prob, fg prob)]
    - rpn_bbox: [batch, num_anchors, (dy, dx, log(dh), log(dw))]
    - anchors: [batch, num_anchors, (y1, x1, y2, x2)] anchors in normalized coordinates
        
    Returns:
    - Proposals in normalized coordinates [batch, rois, (y1, x1, y2, x2)]
    """

    def __init__(self, proposal_count, nms_threshold, **kwargs):
        super(ProposalLayer, self).__init__(**kwargs)
        self.proposal_count = proposal_count
        self.nms_threshold = nms_threshold

    def call(self, inputs):
        scores = inputs[0][:, :, 1]
        deltas = inputs[1]
        deltas = deltas * np.reshape(np.array([0.1, 0.1, 0.2, 0.2]), [1, 1, 4])
        anchors = inputs[2]

        pre_nms_limit = tf.minimum(6000, tf.shape(anchors)[1])
        ix = tf.nn.top_k(scores, pre_nms_limit, sorted=True,
                         name="top_anchors").indices
        scores = batch_slice([scores, ix], lambda x, y: tf.gather(x, y),batch_size)
        deltas = batch_slice([deltas, ix], lambda x, y: tf.gather(x, y),batch_size)
        pre_nms_anchors = batch_slice([anchors, ix], lambda a, x: tf.gather(a, x),batch_size,
                                    names=["pre_nms_anchors"])

        boxes = batch_slice([pre_nms_anchors, deltas],
                                  lambda x, y: apply_box_deltas_graph(x, y),batch_size,
                                  names=["refined_anchors"])

        window = np.array([0, 0, 1, 1], dtype=np.float32)
        boxes = batch_slice(boxes,
                                  lambda x: clip_boxes_graph(x, window),batch_size,
                                  names=["refined_anchors_clipped"])

        def nms(boxes, scores):
            indices = tf.image.non_max_suppression(
                boxes, scores, self.proposal_count,
                self.nms_threshold, name="rpn_non_max_suppression")
            proposals = tf.gather(boxes, indices)

            padding = tf.maximum(self.proposal_count - tf.shape(proposals)[0], 0)
            proposals = tf.pad(proposals, [(0, padding), (0, 0)])
            return proposals
        proposals = batch_slice([boxes, scores], nms,batch_size)
        return proposals

    def compute_output_shape(self, input_shape):
        return (None, self.proposal_count, 4)


class MaskRCNN():
    """
    Encapsulates the Mask RCNN model functionality.
    """

    def __init__(self, training=None):
        self.keras_model = self.build(training=training)

        
    def build(self, training=None):
        
        input_image = layers.Input(
            shape=[None, None, 3], name="input_image")
        input_image_meta = layers.Input(shape=[93],
                                    name="input_image_meta")

 
        input_rpn_match = layers.Input(
                shape=[None, 1], name="input_rpn_match", dtype=tf.int32)
        input_rpn_bbox = layers.Input(
                shape=[None, 4], name="input_rpn_bbox", dtype=tf.float32)

        
        input_gt_class_ids = layers.Input(
                shape=[None], name="input_gt_class_ids", dtype=tf.int32)

        input_gt_boxes = layers.Input(
                shape=[None, 4], name="input_gt_boxes", dtype=tf.float32)
         
        gt_boxes = layers.Lambda(lambda x: norm_boxes_graph(
                x, tf.keras.backend.shape(input_image)[1:3]))(input_gt_boxes)

        input_gt_masks = layers.Input(
                    shape=[1024,1024, None],
                    name="input_gt_masks", dtype=bool)
               
        _, C2, C3, C4, C5 = resnet(input_image, training)

        """
        Generate FPN layers. This part must output 5 feature map (P2, P3, P4, P5, P6) which are inputs of RPN. 
        Please note that after generating P2~P5, it must pass 3*3 conv with 256 filters and same padding.
        1. generate P5 from C5 with 256 filters 1*1 conv
        2. generate P4 from adding C4 with 256 filters 1*1 conv and 2*2 umsampled P5
        3. generate P3 from adding C3 with 256 filters 1*1 conv and 2*2 umsampled P4
        4. generate P2 from adding C2 with 256 filters 1*1 conv and 2*2 umsampled P3
        5. generate P6 from 1*1 maxpooling of P5 with strides 2
        """
        #### Question (b): your implementation starts here (don't delete this line)
        def _upsample_and(px, cx):
            smooth_cx = layers.Conv2D(256, (1, 1))(cx)
            upsample_px = layers.UpSampling2D(size=(2, 2), interpolation="bilinear")(px)
            return layers.Conv2D(256, (3, 3), padding="same")(upsample_px + smooth_cx)

        P5 = layers.Conv2D(256, (1, 1))(C5)
        P5 = layers.Conv2D(256, (3, 3), padding="same")(P5)
        P4 = _upsample_and(P5, C4)
        P3 = _upsample_and(P4, C3)
        P2 = _upsample_and(P3, C2)
        P6 = layers.MaxPool2D(pool_size=(1, 1), strides=(2, 2))(P5)

        #### Question (b): your implementation ends here (don't delete this line)

        rpn_feature_maps = [P2, P3, P4, P5, P6]
        mrcnn_feature_maps = [P2, P3, P4, P5]
        
        anchors = self.get_anchors(np.array([1024,1024,3]))
       
        anchors = np.broadcast_to(anchors, (batch_size,) + anchors.shape)
        
        anchor_layer = AnchorsLayer(name="anchors")
        anchors = anchor_layer(anchors)
        
        rpn = build_rpn_model(1, 3, 256)
        
        layer_outputs = []  
        for p in rpn_feature_maps:
            layer_outputs.append(rpn([p]))
        output_names = ["rpn_class_logits", "rpn_class", "rpn_bbox"]
        outputs = list(zip(*layer_outputs))
        outputs = [layers.Concatenate(axis=1, name=n)(list(o))
                   for o, n in zip(outputs, output_names)]

        rpn_class_logits, rpn_class, rpn_bbox = outputs
        
        proposal_count = 2000
        nms_threshold = 0.7

        rpn_rois = ProposalLayer(
                proposal_count=proposal_count,
                nms_threshold=nms_threshold)([rpn_class, rpn_bbox, anchors])

        active_class_ids = input_image_meta[:,12:]

        target_rois = rpn_rois

        rois, target_class_ids, target_bbox, target_mask = DetectionTargetLayer()([
                        target_rois, input_gt_class_ids, gt_boxes, input_gt_masks])

        mrcnn_class_logits, mrcnn_class, mrcnn_bbox =\
                    fpn_classifier_graph(rois, mrcnn_feature_maps, input_image_meta,
                                         7, 81,True, fc_layers_size=1024)

        mrcnn_mask = build_fpn_mask_graph(rois, mrcnn_feature_maps,
                                                  input_image_meta,
                                                  14, 81, train_bn=True)


        output_rois = layers.Lambda(lambda x: x * 1)(rois)

        rpn_class_loss = layers.Lambda(lambda x: rpn_class_loss_graph(*x), name="rpn_class_loss")(
                [input_rpn_match, rpn_class_logits])
        rpn_bbox_loss = layers.Lambda(lambda x: rpn_bbox_loss_graph(*x), name="rpn_bbox_loss")(
                [input_rpn_bbox, input_rpn_match, rpn_bbox])
        class_loss = layers.Lambda(lambda x: mrcnn_class_loss_graph(*x), name="mrcnn_class_loss")(
                [target_class_ids, mrcnn_class_logits, active_class_ids])
        bbox_loss = layers.Lambda(lambda x: mrcnn_bbox_loss_graph(*x), name="mrcnn_bbox_loss")(
                [target_bbox, target_class_ids, mrcnn_bbox])
        mask_loss = layers.Lambda(lambda x: mrcnn_mask_loss_graph(*x), name="mrcnn_mask_loss")(
                [target_mask, target_class_ids, mrcnn_mask])

        inputs = [input_image, input_image_meta,
                      input_rpn_match, input_rpn_bbox, input_gt_class_ids, input_gt_boxes, input_gt_masks]

        outputs = [rpn_class_logits, rpn_class, rpn_bbox,
                       mrcnn_class_logits, mrcnn_class, mrcnn_bbox, mrcnn_mask,
                       rpn_rois, output_rois,
                       rpn_class_loss, rpn_bbox_loss, class_loss, bbox_loss, mask_loss]         

        model = tf.keras.models.Model(inputs, outputs)

        return model
    
    def get_anchors(self, image_shape):
        """
        Returns anchor pyramid for the given image size.
        """
        backbone_shapes = np.array(
            [[int(math.ceil(image_shape[0] / stride)),
                int(math.ceil(image_shape[1] / stride))]
                for stride in [4, 8, 16, 32, 64]])
       
        if not hasattr(self, "_anchor_cache"):
                self._anchor_cache = {}
        if not tuple(image_shape) in self._anchor_cache:
                # Generate Anchors
            a = generate_pyramid_anchors(
                    (32, 64, 128, 256, 512),
                    [0.5, 1, 2],
                    backbone_shapes,
                    [4, 8, 16, 32, 64],
                    1)
          
            self.anchors = a
            self._anchor_cache[tuple(image_shape)] = norm_boxes(a, image_shape[:2])
        return self._anchor_cache[tuple(image_shape)]


    
    def train(self, train_dataset, val_dataset, epochs, batch_size, learning_rate, checkpoint_path ):
        """
        Train the model.
        
        Inputs:
        - train_dataset, val_dataset: Training and validation Dataset objects.
        - epochs
        - batch_size
        - learning_rate
        - checkpoint_path
        """
        train_generator = data_generator(train_dataset, shuffle=True,
                                             batch_size=batch_size)
        val_generator = data_generator(val_dataset, shuffle=True,
                                           batch_size=batch_size)

        self.compile(learning_rate, 0.9)

        callbacks = [tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                verbose=0, save_weights_only=True)]

        self.keras_model.fit_generator(
                train_generator,
                epochs=epochs,
                steps_per_epoch=1000,
                callbacks=callbacks,
                validation_data=val_generator,
                validation_steps=50,
                
            )


    def compile(self, learning_rate, momentum):
        """
        Gets the model ready for training. Adds losses, regularization, and
        metrics. Then calls the Keras compile() function.
        """

        optimizer = tf.keras.optimizers.SGD(
            lr=learning_rate, momentum=momentum,
            clipnorm=5)
        
        loss_names = [
            "rpn_class_loss",  "rpn_bbox_loss",
            "mrcnn_class_loss", "mrcnn_bbox_loss", "mrcnn_mask_loss"]
        
        exists_names=[]
        for name in loss_names:
            layer = self.keras_model.get_layer(name)
            
            if name in exists_names:
                continue

            
            exists_names.append(name)
            loss = (
                 tf.reduce_mean(input_tensor=layer.output, keepdims=True))
            self.keras_model.add_loss(loss)
        
        reg_losses = [
            tf.keras.regularizers.l2(0.0001)(w) / tf.cast(tf.size(input=w), tf.float32)
            for w in self.keras_model.trainable_weights
            if 'gamma' not in w.name and 'beta' not in w.name]
        self.keras_model.add_loss(lambda: tf.add_n(reg_losses))

        self.keras_model.compile(
            optimizer=optimizer,
            experimental_run_tf_function=False,
            loss=[None] * len(self.keras_model.outputs))

        for name in loss_names:

            if name in self.keras_model.metrics_names:
                continue
            layer = self.keras_model.get_layer(name)
            self.keras_model.metrics_names.append(name)
            loss = (
                tf.reduce_mean(input_tensor=layer.output, keepdims=True))
            self.keras_model.add_metric(loss, name=name, aggregation='mean')

class Dataset(object):
    """
    The base class for dataset classes.
    """

    def __init__(self, class_map=None):
        self._image_ids = []
        self.image_info = []
        self.class_info = [{"source": "", "id": 0, "name": "BG"}]
        self.source_class_ids = {}

    def add_class(self, source, class_id, class_name):
        assert "." not in source, "Source name cannot contain a dot"
        for info in self.class_info:
            if info['source'] == source and info["id"] == class_id:
                return
        self.class_info.append({
            "source": source,
            "id": class_id,
            "name": class_name,
        })

    def add_image(self, source, image_id, path, **kwargs):
        image_info = {
            "id": image_id,
            "source": source,
            "path": path,
        }
        image_info.update(kwargs)
        self.image_info.append(image_info)

    def image_reference(self, image_id):
        
        return ""

    def prepare(self, class_map=None):
        """
        Prepares the Dataset class for use.
        """

        def clean_name(name):
            """
            Returns a shorter version of object names for cleaner display.
            """
            return ",".join(name.split(",")[:1])

        self.num_classes = len(self.class_info)
        self.class_ids = np.arange(self.num_classes)
        self.class_names = [clean_name(c["name"]) for c in self.class_info]
        self.num_images = len(self.image_info)
        self._image_ids = np.arange(self.num_images)

        self.class_from_source_map = {"{}.{}".format(info['source'], info['id']): id
                                      for info, id in zip(self.class_info, self.class_ids)}
        self.image_from_source_map = {"{}.{}".format(info['source'], info['id']): id
                                      for info, id in zip(self.image_info, self.image_ids)}

        self.sources = list(set([i['source'] for i in self.class_info]))
        self.source_class_ids = {}

        for source in self.sources:
            self.source_class_ids[source] = []

            for i, info in enumerate(self.class_info):
                if i == 0 or source == info['source']:
                    self.source_class_ids[source].append(i)

    def map_source_class_id(self, source_class_id):
        """
        Takes a source class ID and returns the int class ID assigned to it.
        """
        return self.class_from_source_map[source_class_id]

    def get_source_class_id(self, class_id, source):
        """
        Map an internal class ID to the corresponding class ID in the source dataset.
        """
        info = self.class_info[class_id]
        assert info['source'] == source
        return info['id']

    @property
    def image_ids(self):
        return self._image_ids

    def source_image_link(self, image_id):
        """
        Returns the path or URL to the image.
        Override this to return a URL to the image if it's available online for easy
        debugging.
        """
        return self.image_info[image_id]["path"]

    def load_image(self, image_id):
        """
        Load the specified image and return a [H,W,3] Numpy array.
        """
        image = skimage.io.imread(self.image_info[image_id]['path'])
        if image.ndim != 3:
            image = skimage.color.gray2rgb(image)

        if image.shape[-1] == 4:
            image = image[..., :3]
        return image

    def load_mask(self, image_id):
        """
        Load instance masks for the given image.
        Different datasets use different ways to store masks. Override this
        method to load instance masks and return them in the form of am
        array of binary masks of shape [height, width, instances].
        """
        logging.warning("You are using the default load_mask(), maybe you need to define your own one.")
        mask = np.empty([0, 0, 0])
        class_ids = np.empty([0], np.int32)
        return mask, class_ids

    
class CocoDataset(Dataset):
    def load_coco(self, dataset_dir, subset, year=2017, class_ids=None,
                  class_map=None, return_coco=False):
        """
        Load a subset of the COCO dataset.
        
        Inputs:
        - dataset_dir: The root directory of the COCO dataset.
        - subset: What to load (train, val, minival, valminusminival)
        - year: What dataset year to load (2014, 2017)
        - class_ids: If provided, only loads images that have the given classes.
        - class_map: TODO: Not implemented yet. Supports maping classes from
            different datasets to the same class ID.
        - return_coco: If True, returns the COCO object.
        """

        
        coco = COCO("{}/annotations/instances_{}{}.json".format(dataset_dir, subset, year))
        if subset == "minival" or subset == "valminusminival":
            subset = "val"
        image_dir = "{}/{}{}".format(dataset_dir, subset, year)

        if not class_ids:
            class_ids = sorted(coco.getCatIds())

        if class_ids:
            image_ids = []
            for id in class_ids:
                image_ids.extend(list(coco.getImgIds(catIds=[id])))
            image_ids = list(set(image_ids))
        else:
            image_ids = list(coco.imgs.keys())

        for i in class_ids:
            self.add_class("coco", i, coco.loadCats(i)[0]["name"])

        for i in image_ids:
            self.add_image(
                "coco", image_id=i,
                path=os.path.join(image_dir, coco.imgs[i]['file_name']),
                width=coco.imgs[i]["width"],
                height=coco.imgs[i]["height"],
                annotations=coco.loadAnns(coco.getAnnIds(
                    imgIds=[i], catIds=class_ids, iscrowd=None)))
        if return_coco:
            return coco

    def load_mask(self, image_id):
        """
        Load instance masks for the given image.
        Different datasets use different ways to store masks. This
        function converts the different mask format to one format
        in the form of a bitmap [height, width, instances].
        
        Returns:
        - masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        - class_ids: a 1D array of class IDs of the instance masks.
        """
        image_info = self.image_info[image_id]
        if image_info["source"] != "coco":
            return super(CocoDataset, self).load_mask(image_id)

        instance_masks = []
        class_ids = []
        annotations = self.image_info[image_id]["annotations"]

        for annotation in annotations:
            class_id = self.map_source_class_id(
                "coco.{}".format(annotation['category_id']))
            if class_id:
                m = self.annToMask(annotation, image_info["height"],
                                   image_info["width"])

                if m.max() < 1:
                    continue

                if annotation['iscrowd']:

                    class_id *= -1

                    if m.shape[0] != image_info["height"] or m.shape[1] != image_info["width"]:
                        m = np.ones([image_info["height"], image_info["width"]], dtype=bool)
                instance_masks.append(m)
                class_ids.append(class_id)


        if class_ids:
            mask = np.stack(instance_masks, axis=2).astype(np.bool)
            class_ids = np.array(class_ids, dtype=np.int32)
            return mask, class_ids
        else:

            return super(CocoDataset, self).load_mask(image_id)

    def image_reference(self, image_id):
        """
        Return a link to the image in the COCO Website.
        """
        info = self.image_info[image_id]
        if info["source"] == "coco":
            return "http://cocodataset.org/#explore?id={}".format(info["id"])
        else:
            super(CocoDataset, self).image_reference(image_id)

    def annToRLE(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        :return: binary mask (numpy 2D array)
        """
        segm = ann['segmentation']
        if isinstance(segm, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(segm, height, width)
            rle = maskUtils.merge(rles)
        elif isinstance(segm['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(segm, height, width)
        else:
            # rle
            rle = ann['segmentation']
        return rle

    def annToMask(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
        :return: binary mask (numpy 2D array)
        """
        rle = self.annToRLE(ann, height, width)
        m = maskUtils.decode(rle)
        return m
        
def mold_image(images):
    """
    Expects an RGB image (or array of images) and subtracts
    the mean pixel and converts it to float. Expects image
    colors in RGB order.
    """
    return images.astype(np.float32) - np.array([123.7, 116.8, 103.9])


def resize(image, output_shape, order=1, mode='constant', cval=0, clip=True,
           preserve_range=False, anti_aliasing=False, anti_aliasing_sigma=None):
    """
    A wrapper for Scikit-Image resize().
    Scikit-Image generates warnings on every call to resize() if it doesn't
    receive the right parameters. The right parameters depend on the version
    of skimage. This solves the problem by using different parameters per
    version. And it provides a central place to control resizing defaults.
    """
    if LooseVersion(skimage.__version__) >= LooseVersion("0.14"):

        return skimage.transform.resize(
            image, output_shape,
            order=order, mode=mode, cval=cval, clip=clip,
            preserve_range=preserve_range, anti_aliasing=anti_aliasing,
            anti_aliasing_sigma=anti_aliasing_sigma)
    else:
        return skimage.transform.resize(
            image, output_shape,
            order=order, mode=mode, cval=cval, clip=clip,
            preserve_range=preserve_range)

def compute_iou(box, boxes, box_area, boxes_area):
    """
    Calculates IoU of the given box with the array of the given boxes.
    
    Inputs:
    - box: 1D vector [y1, x1, y2, x2]
    - boxes: [boxes_count, (y1, x1, y2, x2)]
    - box_area: float. the area of 'box'
    - boxes_area: array of length boxes_count.
    """
    y1 = np.maximum(box[0], boxes[:, 0])
    y2 = np.minimum(box[2], boxes[:, 2])
    x1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[3], boxes[:, 3])
    intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    union = box_area + boxes_area[:] - intersection[:]
    iou = intersection / union
    return iou


def compute_overlaps(boxes1, boxes2):
    """
    Computes IoU overlaps between two sets of boxes.
    boxes1, boxes2: [N, (y1, x1, y2, x2)].
    For better performance, pass the largest set first and the smaller second.
    """

    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])


    overlaps = np.zeros((boxes1.shape[0], boxes2.shape[0]))
    for i in range(overlaps.shape[1]):
        box2 = boxes2[i]
        overlaps[:, i] = compute_iou(box2, boxes1, area2[i], area1)
    return overlaps


def build_rpn_targets(image_shape, anchors, gt_class_ids, gt_boxes):
    """
    Given the anchors and GT boxes, compute overlaps and identify positive
    anchors and deltas to refine them to match their corresponding GT boxes.
    
    Inputs:
    - image_shape
    - anchors: [num_anchors, (y1, x1, y2, x2)]
    - gt_class_ids: [num_gt_boxes] Integer class IDs.
    - gt_boxes: [num_gt_boxes, (y1, x1, y2, x2)]
    
    Returns:
    - rpn_match: [N] (int32) matches between anchors and GT boxes.
               1 = positive anchor, -1 = negative anchor, 0 = neutral
    - rpn_bbox: [N, (dy, dx, log(dh), log(dw))] Anchor bbox deltas.
    """
   
    rpn_match = np.zeros([anchors.shape[0]], dtype=np.int32)
    rpn_bbox = np.zeros((256, 4))

    crowd_ix = np.where(gt_class_ids < 0)[0]
    if crowd_ix.shape[0] > 0:
        non_crowd_ix = np.where(gt_class_ids > 0)[0]
        crowd_boxes = gt_boxes[crowd_ix]
        gt_class_ids = gt_class_ids[non_crowd_ix]
        gt_boxes = gt_boxes[non_crowd_ix]
        crowd_overlaps = compute_overlaps(anchors, crowd_boxes)
        crowd_iou_max = np.amax(crowd_overlaps, axis=1)
        no_crowd_bool = (crowd_iou_max < 0.001)
    else:
        no_crowd_bool = np.ones([anchors.shape[0]], dtype=bool)

    overlaps = compute_overlaps(anchors, gt_boxes)

    anchor_iou_argmax = np.argmax(overlaps, axis=1)
    anchor_iou_max = overlaps[np.arange(overlaps.shape[0]), anchor_iou_argmax]
    rpn_match[(anchor_iou_max < 0.3) & (no_crowd_bool)] = -1

    gt_iou_argmax = np.argwhere(overlaps == np.max(overlaps, axis=0))[:,0]
    rpn_match[gt_iou_argmax] = 1

    rpn_match[anchor_iou_max >= 0.7] = 1


    ids = np.where(rpn_match == 1)[0]
    extra = len(ids) - (256 // 2)
    if extra > 0:

        ids = np.random.choice(ids, extra, replace=False)
        rpn_match[ids] = 0

    ids = np.where(rpn_match == -1)[0]
    extra = len(ids) - (256 -
                        np.sum(rpn_match == 1))
    if extra > 0:

        ids = np.random.choice(ids, extra, replace=False)
        rpn_match[ids] = 0

    ids = np.where(rpn_match == 1)[0]
    ix = 0  

    for i, a in zip(ids, anchors[ids]):

        gt = gt_boxes[anchor_iou_argmax[i]]

        gt_h = gt[2] - gt[0]
        gt_w = gt[3] - gt[1]
        gt_center_y = gt[0] + 0.5 * gt_h
        gt_center_x = gt[1] + 0.5 * gt_w

        a_h = a[2] - a[0]
        a_w = a[3] - a[1]
        a_center_y = a[0] + 0.5 * a_h
        a_center_x = a[1] + 0.5 * a_w

        rpn_bbox[ix] = [
            (gt_center_y - a_center_y) / a_h,
            (gt_center_x - a_center_x) / a_w,
            np.log(gt_h / a_h),
            np.log(gt_w / a_w),
        ]

        rpn_bbox[ix] /= np.array([0.1, 0.1, 0.2, 0.2])
        ix += 1

    return rpn_match, rpn_bbox


def compose_image_meta(image_id, original_image_shape, image_shape,
                       window, scale, active_class_ids):
    """
    Takes attributes of an image and puts them in one 1D array.
    """
    meta = np.array(
        [image_id] +                  # size=1
        list(original_image_shape) +  # size=3
        list(image_shape) +           # size=3
        list(window) +                # size=4 (y1, x1, y2, x2) in image cooredinates
        [scale] +                     # size=1
        list(active_class_ids)        # size=num_classes
    )
    return meta

def resize_image(image, min_dim=None, max_dim=None, min_scale=None, mode="square"):
    """
    Resizes an image keeping the aspect ratio unchanged.
    
    Inputs:
    - min_dim: if provided, resizes the image such that it's smaller
        dimension == min_dim
    - max_dim: if provided, ensures that the image longest side doesn't
        exceed this value.
    - min_scale: if provided, ensure that the image is scaled up by at least
        this percent even if min_dim doesn't require it.
    - mode: Resizing mode.
        
    Returns:
    - image: the resized image
    - window: (y1, x1, y2, x2). If max_dim is provided, padding might
        be inserted in the returned image. If so, this window is the
        coordinates of the image part of the full image (excluding
        the padding). The x2, y2 pixels are not included.
    - scale: The scale factor used to resize the image
    - padding: Padding added to the image [(top, bottom), (left, right), (0, 0)]
    """

    image_dtype = image.dtype

    h, w = image.shape[:2]
    window = (0, 0, h, w)
    scale = 1
    padding = [(0, 0), (0, 0), (0, 0)]
    crop = None

    if mode == "none":
        return image, window, scale, padding, crop


    if min_dim:

        scale = max(1, min_dim / min(h, w))
    if min_scale and scale < min_scale:
        scale = min_scale

    if max_dim and mode == "square":
        image_max = max(h, w)
        if round(image_max * scale) > max_dim:
            scale = max_dim / image_max


    if scale != 1:
        image = resize(image, (round(h * scale), round(w * scale)),
                       preserve_range=True)


    if mode == "square":

        h, w = image.shape[:2]
        top_pad = (max_dim - h) // 2
        bottom_pad = max_dim - h - top_pad
        left_pad = (max_dim - w) // 2
        right_pad = max_dim - w - left_pad
        padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
        image = np.pad(image, padding, mode='constant', constant_values=0)
        window = (top_pad, left_pad, h + top_pad, w + left_pad)
    elif mode == "pad64":
        h, w = image.shape[:2]

        assert min_dim % 64 == 0, "Minimum dimension must be a multiple of 64"

        if h % 64 > 0:
            max_h = h - (h % 64) + 64
            top_pad = (max_h - h) // 2
            bottom_pad = max_h - h - top_pad
        else:
            top_pad = bottom_pad = 0

        if w % 64 > 0:
            max_w = w - (w % 64) + 64
            left_pad = (max_w - w) // 2
            right_pad = max_w - w - left_pad
        else:
            left_pad = right_pad = 0
        padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
        image = np.pad(image, padding, mode='constant', constant_values=0)
        window = (top_pad, left_pad, h + top_pad, w + left_pad)
    elif mode == "crop":
        h, w = image.shape[:2]
        y = random.randint(0, (h - min_dim))
        x = random.randint(0, (w - min_dim))
        crop = (y, x, min_dim, min_dim)
        image = image[y:y + min_dim, x:x + min_dim]
        window = (0, 0, min_dim, min_dim)
    else:
        raise Exception("Mode {} not supported".format(mode))
    return image.astype(image_dtype), window, scale, padding, crop

def resize_mask(mask, scale, padding, crop=None):
    """
    Resizes a mask using the given scale and padding.
    """
    # Suppress warning from scipy 0.13.0, the output shape of zoom() is
    # calculated with round() instead of int()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mask = scipy.ndimage.zoom(mask, zoom=[scale, scale, 1], order=0)
    if crop is not None:
        y, x, h, w = crop
        mask = mask[y:y + h, x:x + w]
    else:
        mask = np.pad(mask, padding, mode='constant', constant_values=0)
    return mask

def extract_bboxes(mask):
    """
    Compute bounding boxes from masks.
    
    Inputs:
    - mask: [height, width, num_instances]. Mask pixels are either 1 or 0.
    Returns: 
    - bbox array: [num_instances, (y1, x1, y2, x2)].
    """
    boxes = np.zeros([mask.shape[-1], 4], dtype=np.int32)
    for i in range(mask.shape[-1]):
        m = mask[:, :, i]
        # Bounding box.
        horizontal_indicies = np.where(np.any(m, axis=0))[0]
        vertical_indicies = np.where(np.any(m, axis=1))[0]
        if horizontal_indicies.shape[0]:
            x1, x2 = horizontal_indicies[[0, -1]]
            y1, y2 = vertical_indicies[[0, -1]]
            # x2 and y2 should not be part of the box. Increment by 1.
            x2 += 1
            y2 += 1
        else:
            # No mask for this instance. Might happen due to
            # resizing or cropping. Set bbox to zeros
            x1, x2, y1, y2 = 0, 0, 0, 0
        boxes[i] = np.array([y1, x1, y2, x2])
    return boxes.astype(np.int32)


def load_image_gt(dataset, image_id, augment=False, augmentation=None,
                  use_mini_mask=False):
    """
    Load and return ground truth data for an image (image, mask, bounding boxes).
    
    Inputs:
    - dataset
    - image_id
    
    Returns:
    - image: [height, width, 3]
    - shape: the original shape of the image before resizing and cropping.
    - class_ids: [instance_count] Integer class IDs
    - bbox: [instance_count, (y1, x1, y2, x2)]
    - mask: [height, width, instance_count]. The height and width are those
        of the image unless use_mini_mask is True, in which case they are
        defined in MINI_MASK_SHAPE.
    """
    image = dataset.load_image(image_id)
    mask, class_ids = dataset.load_mask(image_id)
    original_shape = image.shape
    image, window, scale, padding, crop = resize_image(
        image,
        min_dim=800,
        min_scale=0,
        max_dim=1024,
        mode="square")
    mask = resize_mask(mask, scale, padding, crop)


    _idx = np.sum(mask, axis=(0, 1)) > 0
    mask = mask[:, :, _idx]
    class_ids = class_ids[_idx]

    bbox = extract_bboxes(mask)


    active_class_ids = np.zeros([dataset.num_classes], dtype=np.int32)
    source_class_ids = dataset.source_class_ids[dataset.image_info[image_id]["source"]]
    active_class_ids[source_class_ids] = 1


    image_meta = compose_image_meta(image_id, original_shape, image.shape,
                                    window, scale, active_class_ids)

    return image, image_meta, class_ids, bbox, mask

def data_generator(dataset, shuffle=True, augment=False, augmentation=None,
                   random_rois=0, batch_size=1, detection_targets=False,
                   no_augmentation_sources=None):
    """
    A generator that returns images and corresponding target class ids,
    bounding box deltas, and masks.
    
    Inputs:
    - dataset: The Dataset object to pick data from
    - shuffle: If True, shuffles the samples before every epoch
    - batch_size: How many images to return in each call
    
    Returns:
    - a Python generator. Upon calling next() on it, the
    generator returns two lists, inputs and outputs. The contents
    of the lists differs depending on the received arguments:
        inputs list:
        - images: [batch, H, W, C]
        - image_meta: [batch, (meta data)] Image details. See compose_image_meta()
        - rpn_match: [batch, N] Integer (1=positive anchor, -1=negative, 0=neutral)
        - rpn_bbox: [batch, N, (dy, dx, log(dh), log(dw))] Anchor bbox deltas.
        - gt_class_ids: [batch, MAX_GT_INSTANCES] Integer class IDs
        - gt_boxes: [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)]
        - gt_masks: [batch, height, width, MAX_GT_INSTANCES]. The height and width
                    are those of the image unless use_mini_mask is True, in which
                    case they are defined in MINI_MASK_SHAPE.
        outputs list: Usually empty in regular training. 
    """
    b = 0  
    image_index = -1
    image_ids = np.copy(dataset.image_ids)
    error_count = 0
    no_augmentation_sources = no_augmentation_sources or []

   
    backbone_shapes = np.array(
        [[int(math.ceil(1024 / stride)),
          int(math.ceil(1024 / stride))] for stride in [4, 8, 16, 32, 64]])
    anchors = generate_pyramid_anchors((32, 64, 128, 256, 512),
                                             [0.5, 1, 2],
                                             backbone_shapes,
                                             [4, 8, 16, 32, 64],
                                             1)


    while True:
        try:
            
            image_index = (image_index + 1) % len(image_ids)
            if shuffle and image_index == 0:
                np.random.shuffle(image_ids)

            
            image_id = image_ids[image_index]

            
            if dataset.image_info[image_id]['source'] in no_augmentation_sources:
                image, image_meta, gt_class_ids, gt_boxes, gt_masks = \
                load_image_gt(dataset, image_id, augment=augment,
                              augmentation=None,
                              use_mini_mask=False)
            else:
                image, image_meta, gt_class_ids, gt_boxes, gt_masks = \
                    load_image_gt(dataset, image_id, augment=augment,
                                augmentation=augmentation,
                                use_mini_mask=False)

            
            if not np.any(gt_class_ids > 0):
                continue

            
            rpn_match, rpn_bbox = build_rpn_targets(image.shape, anchors,
                                                    gt_class_ids, gt_boxes)


            if b == 0:
                batch_image_meta = np.zeros(
                    (batch_size,) + image_meta.shape, dtype=image_meta.dtype)
                batch_rpn_match = np.zeros(
                    [batch_size, anchors.shape[0], 1], dtype=rpn_match.dtype)
                batch_rpn_bbox = np.zeros(
                    [batch_size, 256, 4], dtype=rpn_bbox.dtype)
                batch_images = np.zeros(
                    (batch_size,) + image.shape, dtype=np.float32)
                batch_gt_class_ids = np.zeros(
                    (batch_size,100), dtype=np.int32)
                batch_gt_boxes = np.zeros(
                    (batch_size, 100, 4), dtype=np.int32)
                batch_gt_masks = np.zeros(
                    (batch_size, gt_masks.shape[0], gt_masks.shape[1],
                     100), dtype=gt_masks.dtype)
                
            if gt_boxes.shape[0] > 100:
                ids = np.random.choice(
                    np.arange(gt_boxes.shape[0]), 100, replace=False)
                gt_class_ids = gt_class_ids[ids]
                gt_boxes = gt_boxes[ids]
                gt_masks = gt_masks[:, :, ids]

            batch_image_meta[b] = image_meta
            batch_rpn_match[b] = rpn_match[:, np.newaxis]
            batch_rpn_bbox[b] = rpn_bbox
            batch_images[b] = mold_image(image.astype(np.float32))
            batch_gt_class_ids[b, :gt_class_ids.shape[0]] = gt_class_ids
            batch_gt_boxes[b, :gt_boxes.shape[0]] = gt_boxes
            batch_gt_masks[b, :, :, :gt_masks.shape[-1]] = gt_masks
            
            b += 1

            if b >= batch_size:
                inputs = [batch_images, batch_image_meta, batch_rpn_match, batch_rpn_bbox,
                          batch_gt_class_ids, batch_gt_boxes, batch_gt_masks]
                outputs = []

                
                yield inputs, outputs

                b = 0
        except (GeneratorExit, KeyboardInterrupt):
            raise

def generate_anchors(scales, ratios, shape, feature_stride, anchor_stride):
    """
    Generate anchors in feature space
    
    Inputs:
    - scales: 1D array of anchor sizes in pixels. Example: [32, 64, 128]
    - ratios: 1D array of anchor ratios of width/height. Example: [0.5, 1, 2]
    - shape: [height, width] spatial shape of the feature map over which
            to generate anchors.
    - feature_stride: Stride of the feature map relative to the image in pixels.
    - anchor_stride: Stride of anchors on the feature map. For example, if the
        value is 2 then generate anchors for every other feature map pixel.
        
    Returns:
    - boxes: anchor coordinates (y1, x1, y2, x2)
    
    """
    scales, ratios = np.meshgrid(np.array(scales), np.array(ratios))
    scales = scales.flatten()
    ratios = ratios.flatten()

    heights = scales / np.sqrt(ratios)
    widths = scales * np.sqrt(ratios)

    shifts_y = np.arange(0, shape[0], anchor_stride) * feature_stride
    shifts_x = np.arange(0, shape[1], anchor_stride) * feature_stride
    shifts_x, shifts_y = np.meshgrid(shifts_x, shifts_y)

    box_widths, box_centers_x = np.meshgrid(widths, shifts_x)
    box_heights, box_centers_y = np.meshgrid(heights, shifts_y)

    box_centers = np.stack(
        [box_centers_y, box_centers_x], axis=2).reshape([-1, 2])
    box_sizes = np.stack([box_heights, box_widths], axis=2).reshape([-1, 2])

    boxes = np.concatenate([box_centers - 0.5 * box_sizes,
                            box_centers + 0.5 * box_sizes], axis=1)
    return boxes


def generate_pyramid_anchors(scales, ratios, feature_shapes, feature_strides,
                             anchor_stride):
    """
    Generate anchors at different levels of a feature pyramid. Each scale
    is associated with a level of the pyramid, but each ratio is used in
    all levels of the pyramid.
    
    Inputs:
    - scales: 1D array of anchor sizes in pixels. Example: [32, 64, 128]
    - ratios: 1D array of anchor ratios of width/height. Example: [0.5, 1, 2]
    - shape: [height, width] spatial shape of the feature map over which
            to generate anchors.
    - feature_stride: Stride of the feature map relative to the image in pixels.
    - anchor_stride: Stride of anchors on the feature map. For example, if the
        value is 2 then generate anchors for every other feature map pixel.
        
    Returns:
    anchors: [N, (y1, x1, y2, x2)]. All generated anchors in one array. Sorted
        with the same order of the given scales. So, anchors of scale[0] come
        first, then anchors of scale[1], and so on.
    """

    anchors = []
    for i in range(len(scales)):
        anchors.append(generate_anchors(scales[i], ratios, feature_shapes[i],
                                        feature_strides[i], anchor_stride))
    return np.concatenate(anchors, axis=0)


def main():
    global data_path
    global ckpt_path
    global epochs
    global learning_rate
    global batch_size

    dataset_train = CocoDataset()

    dataset_train.load_coco(data_path, "train", year=2017)

    dataset_train.prepare()

    dataset_val = CocoDataset()

    dataset_val.load_coco(data_path, 'val', year=2017)

    dataset_val.prepare()

    model2 = MaskRCNN(training=True)

    model2.train(dataset_train, dataset_val, epochs, batch_size, learning_rate,  ckpt_path)

if __name__ == '__main__':
    main()

