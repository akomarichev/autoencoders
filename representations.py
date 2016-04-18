import numpy as np

import helper
import model


def get_representations_k_deep_sparse(theta, images, patch_size, image_size, N, K):

    numberOfPatches_l1 = image_size // patch_size
    numberOfPatches_l2 = image_size // (2 * patch_size)
    numberOfPatches_l3 = image_size // (4 * patch_size)

    W1 = theta['W1']
    W2 = theta['W2']
    W3 = theta['W3']
    W4 = theta['W4']

    layerwise_outputs = model.feed_forward(theta, images, patch_size, image_size, N, K)

    z2 = layerwise_outputs['z2']
    z2_mask = layerwise_outputs['z2_mask']
    z3 = layerwise_outputs['z3']
    z3_mask = layerwise_outputs['z3_mask']
    z4 = layerwise_outputs['z4']
    z4_mask = layerwise_outputs['z4_mask']
    h = layerwise_outputs['h']

    helper.save_image_l(z2, N, "repr_hl1_without_mask")
    helper.save_image_l(z2 * z2_mask, N, "repr_hl1_with_mask")
    helper.save_image_l(z3, N, "repr_hl2_without_mask")
    helper.save_image_l(z3 * z3_mask, N, "repr_hl2_with_mask")
    helper.save_image_l(z4, N, "repr_hl3_without_mask")
    helper.save_image_l(z4 * z4_mask, N, "repr_hl3_with_mask")
    helper.save_image_l(images, N, "original")
    helper.save_image_l(h, N, "output")

    helper.save_image_w(W1, numberOfPatches_l1, patch_size, "W1")
    helper.save_image_w(W2, numberOfPatches_l2, 2 * patch_size, "W2")
    helper.save_image_w(W3, numberOfPatches_l3, 4 * patch_size, "W3")
    helper.save_image_l(W4.reshape(image_size, image_size, image_size ** 2), image_size ** 2, "W4")
