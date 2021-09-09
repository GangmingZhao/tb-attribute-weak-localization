from .builder import DATASETS
from .coco import CocoDataset

import torch
from mmcv.parallel import DataContainer as DC

@DATASETS.register_module()
class TBDataset(CocoDataset):

    CLASSES = ('ActiveTuberculosis', 'ObsoletePulmonaryTuberculosis')

@DATASETS.register_module()
class TBDataset_ATTR(CocoDataset):

    CLASSES = ('ActiveTuberculosis', 'ObsoletePulmonaryTuberculosis')

    def prepare_train_img(self, idx):
        """Get training data and annotations after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys \
                introduced by pipeline.
        """
        
        data = super().prepare_train_img(idx)
        filename = data['img_metas']._data['ori_filename']
        if 'attr' in filename:
            ss = filename.split('/')[-1].split('.')[0]
            assert ss.count('_') == 8
            ss = [int(s) for s in ss.split('_')[1:]]
            data['gt_attrs'] = DC(torch.LongTensor(ss))
        else:
            data['gt_attrs'] = DC(torch.LongTensor([-1]*8))   
        return data
    
    def _filter_imgs(self, min_size=32):
        """Filter images too small"""
        valid_inds = []
        # obtain images that contain annotation
        ids_with_ann = set(_['image_id'] for _ in self.coco.anns.values())
        # obtain images that contain annotations of the required categories
        ids_in_cat = set()
        for i, class_id in enumerate(self.cat_ids):
            ids_in_cat |= set(self.coco.cat_img_map[class_id])
        # merge the image id sets of the two conditions and use the merged set
        # to filter out images if self.filter_empty_gt=True
        ids_in_cat &= ids_with_ann

        valid_img_ids = []
        for i, img_info in enumerate(self.data_infos):
            img_id = self.img_ids[i]
            #if self.filter_empty_gt and img_id not in ids_in_cat:
            #    continue
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
                valid_img_ids.append(img_id)
        self.img_ids = valid_img_ids
        return valid_inds