import mmcv
import numpy as np
from mmdet.datasets.pipelines.formatting import DefaultFormatBundle, to_tensor
import os.path as osp
from ..builder import ROTATED_PIPELINES
from mmcv.parallel import DataContainer as DC

@ROTATED_PIPELINES.register_module()
class PairedImageDefaultFormatBundle(DefaultFormatBundle):
    """PairImage Default formatting bundle similar with Default formatting bundle.

        It simplifies the pipeline of formatting common fields, including "img", "img_tir"
        "proposals", "gt_bboxes", "gt_labels", "gt_masks" and "gt_semantic_seg".
        These fields are formatted as follows.

        - img: (1)transpose, (2)to tensor, (3)to DataContainer (stack=True)
        - img_tir: (1)transpose, (2)to tensor, (3)to DataContainer (stack=True)
        - proposals: (1)to tensor, (2)to DataContainer
        - gt_bboxes: (1)to tensor, (2)to DataContainer
        - gt_bboxes_ignore: (1)to tensor, (2)to DataContainer
        - gt_labels: (1)to tensor, (2)to DataContainer
        - gt_masks: (1)to tensor, (2)to DataContainer (cpu_only=True)
        - gt_semantic_seg: (1)unsqueeze dim-0 (2)to tensor, \
                           (3)to DataContainer (stack=True)

        Args:
            img_to_float (bool): Whether to force the image to be converted to
                float type. Default: True.
            pad_val (dict): A dict for padding value in batch collating,
                the default value is `dict(img=0, masks=0, seg=255)`.
                Without this argument, the padding value of "gt_semantic_seg"
                will be set to 0 by default, which should be 255.
    """
    def __call__(self, results):
        """Call function to transform and format common fields in results.

        Args:
            results (dict): Result dict contains the data to convert.

        Returns:
            dict: The result dict contains the data that is formatted with \
                default bundle.
        """

        if 'img' in results:
            img = results['img']
            if self.img_to_float is True and img.dtype == np.uint8:
                # Normally, image is of uint8 type without normalization.
                # At this time, it needs to be forced to be converted to
                # flot32, otherwise the model training and inference
                # will be wrong. Only used for YOLOX currently .
                img = img.astype(np.float32)
            # add default meta keys
            results = self._add_default_meta_keys(results)
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            img = np.ascontiguousarray(img.transpose(2, 0, 1))
            results['img'] = DC(
                to_tensor(img), padding_value=self.pad_val['img'], stack=True)

        # add imr_tir
        if 'img_tir' in results:
            img_tir = results['img_tir']
            if self.img_to_float is True and img_tir.dtype == np.uint8:
                # Normally, image is of uint8 type without normalization.
                # At this time, it needs to be forced to be converted to
                # flot32, otherwise the model training and inference
                # will be wrong. Only used for YOLOX currently .
                img_tir = img_tir.astype(np.float32)
            # add default meta keys
            results = self._add_default_meta_keys(results)
            if len(img_tir.shape) < 3:
                img_tir = np.expand_dims(img_tir, -1)
            img_tir = np.ascontiguousarray(img_tir.transpose(2, 0, 1))
            results['img_tir'] = DC(
                to_tensor(img_tir), padding_value=self.pad_val['img'], stack=True)

        for key in ['proposals', 'gt_bboxes', 'gt_bboxes_ignore', 'gt_labels']:
            if key not in results:
                continue
            results[key] = DC(to_tensor(results[key]))
        if 'gt_masks' in results:
            results['gt_masks'] = DC(
                results['gt_masks'],
                padding_value=self.pad_val['masks'],
                cpu_only=True)
        if 'gt_semantic_seg' in results:
            results['gt_semantic_seg'] = DC(
                to_tensor(results['gt_semantic_seg'][None, ...]),
                padding_value=self.pad_val['seg'],
                stack=True)
        return results

    def _add_default_meta_keys(self, results):
        """Add default meta keys.

        We set default meta keys including `pad_shape`, `scale_factor` and
        `img_norm_cfg` to avoid the case where no `Resize`, `Normalize` and
        `Pad` are implemented during the whole pipeline.

        Args:
            results (dict): Result dict contains the data to convert.

        Returns:
            results (dict): Updated result dict contains the data to convert.
        """
        img = results['img']
        results.setdefault('pad_shape', img.data.shape)
        results.setdefault('scale_factor', 1.0)
        num_channels = 1 if len(img.data.shape) < 3 else img.data.shape[2]
        results.setdefault(
            'img_norm_cfg',
            dict(
                mean=np.zeros(num_channels, dtype=np.float32),
                std=np.ones(num_channels, dtype=np.float32),
                to_rgb=False))
        return results