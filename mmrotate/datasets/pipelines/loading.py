# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import numpy as np
from mmdet.datasets.pipelines import LoadImageFromFile
import os.path as osp
from ..builder import ROTATED_PIPELINES


@ROTATED_PIPELINES.register_module()
class LoadPatchFromImage(LoadImageFromFile):
    """Load an patch from the huge image.

    Similar with :obj:`LoadImageFromFile`, but only reserve a patch of
    ``results['img']`` according to ``results['win']``.
    """

    def __call__(self, results):
        """Call functions to add image meta information.

        Args:
            results (dict): Result dict with image in ``results['img']``.

        Returns:
            dict: The dict contains the loaded patch and meta information.
        """

        img = results['img']
        x_start, y_start, x_stop, y_stop = results['win']
        width = x_stop - x_start
        height = y_stop - y_start

        patch = img[y_start:y_stop, x_start:x_stop]
        if height > patch.shape[0] or width > patch.shape[1]:
            patch = mmcv.impad(patch, shape=(height, width))

        if self.to_float32:
            patch = patch.astype(np.float32)

        results['filename'] = None
        results['ori_filename'] = None
        results['img'] = patch
        results['img_shape'] = patch.shape
        results['ori_shape'] = patch.shape
        results['img_fields'] = ['img']
        return results


@ROTATED_PIPELINES.register_module()
class LoadPairedImageFromFile(LoadImageFromFile):
    """Load an patch from the huge image.

    Similar with :obj:`LoadImageFromFile`, but load pair images of
    ``results['img']`` according to ``results['img_tir']``.
    """
    def __call__(self, results):
        """Call functions to load image and get image meta information.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results['img_prefix'] is not None:
            filename_vis = osp.join(results['img_prefix'],
                                results['img_info']['filename'])
            filename_tir = osp.join(results['img_prefix'],
                                    results['img_info']['filename_tir'])
        else:
            filename_vis = results['img_info']['filename']
            filename_tir = results['img_info']['filename_tir']

        vis_img_bytes = self.file_client.get(filename_vis)
        vis_img = mmcv.imfrombytes(vis_img_bytes, flag=self.color_type, channel_order=self.channel_order)

        tir_img_bytes = self.file_client.get(filename_tir)
        tir_img = mmcv.imfrombytes(tir_img_bytes, flag=self.color_type, channel_order=self.channel_order)

        if self.to_float32:
            vis_img = vis_img.astype(np.float32)
            tir_img = tir_img.astype(np.float32)

        results['filename'] = filename_vis
        results['filename_tir'] = filename_tir
        results['ori_filename'] = results['img_info']['filename']
        results['ori_filename_tir'] = results['img_info']['filename_tir']
        results['img'] = vis_img
        results['img_tir'] = tir_img
        results['img_shape'] = vis_img.shape
        results['ori_shape'] = vis_img.shape
        results['img_fields'] = ['img', 'img_tir']
        return results