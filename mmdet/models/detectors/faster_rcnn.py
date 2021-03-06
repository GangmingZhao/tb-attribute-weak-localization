from ..builder import DETECTORS
from .two_stage import TwoStageDetector

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, xavier_init

from ..losses import CrossEntropyLoss

@DETECTORS.register_module()
class FasterRCNN(TwoStageDetector):
    """Implementation of `Faster R-CNN <https://arxiv.org/abs/1506.01497>`_"""

    def __init__(self,
                 backbone,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 pretrained=None):
        super(FasterRCNN, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained)

def channel_shuffle(x, groups=2):
    src_shape = x.shape
    new_shape = list(x.shape)
    new_shape[1] = groups
    new_shape.insert(2, x.size(1) // groups)

    x = x.view(*new_shape)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(*src_shape)
    return x

class AttrFusion(nn.Module):
    
    def __init__(self, 
                 in_channels=256,
                 mid_channels=256,
                 num_classes=8):
        super(AttrFusion, self).__init__()
        
        self.num_classes = num_classes
        self.mlp_f = nn.Linear(in_channels, mid_channels)
        self.mlp_a = nn.Linear(in_channels, mid_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.feat_convs = nn.ModuleList()
        for i in range(5):
            conv = ConvModule(
                num_classes*in_channels,
                num_classes*in_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                groups = num_classes,
                norm_cfg=dict(type='BN'),
                act_cfg=dict(type='ReLU'),
                inplace=False
            )
            self.feat_convs.append(conv)
        
    def forward(self, feats, attr_feat):
        
        feats = list(feats)
        avg_attr = F.avg_pool2d(attr_feat, (attr_feat.size(2), attr_feat.size(3)), stride=(attr_feat.size(2), attr_feat.size(3)))
        avg_attr = avg_attr.view(avg_attr.size(0), self.num_classes, -1)
        avg_attr = self.mlp_a(avg_attr)
        for i, x in enumerate(feats):
            
            B, C, H, W = x.size()
            
            avg_x = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
            avg_x = self.mlp_f(avg_x.view(B,-1))
            
            attn = (avg_attr @ avg_x.unsqueeze(-1)).squeeze(-1) #B*N
           
            y = F.interpolate(attr_feat, size=(H, W), mode='bilinear', align_corners=True)
            y = self.feat_convs[i](y)
            y = y.view(B, self.num_classes, -1, H, W)
            
            attn = F.softmax(attn,-1).unsqueeze(2).unsqueeze(3).unsqueeze(4).expand_as(y)
            
            y = (y * attn).sum(1)
            
            feats[i] = self.relu(x + y)
        
        return tuple(feats)
            
    

class AttrHead(nn.Module):
    
    def __init__(self,
                 in_channels=2048,
                 out_channels=1024,
                 num_classes=8):
        super(AttrHead, self).__init__()
        
        '''
        self.conv1 = ConvModule(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='ReLU'),
            inplace=False
        )
        self.conv2 = ConvModule(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='ReLU'),
            inplace=False
        )
        
        self.fc = nn.Linear(out_channels, num_classes)
        '''
        out_channels = 256
        self.num_classes = num_classes
        self.conv1 = ConvModule(
            in_channels,
            num_classes*out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='ReLU'),
            inplace=False
        )
        
        self.conv2 = ConvModule(
            num_classes*out_channels,
            num_classes*out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            groups = num_classes,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='ReLU'),
            inplace=False
        )
        
        self.conv3 = ConvModule(
            num_classes*out_channels,
            num_classes*out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            groups = num_classes,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='ReLU'),
            inplace=False
        )
        
        self.fc = nn.Linear(out_channels, 1)
        
        
    def forward(self, x):
        #x = self.conv1(x)
        #x = self.conv2(x)
        #x = F.adaptive_avg_pool2d(x, 1)
        #x = x.view(x.size(0), -1)
        #x = self.fc(x)
        
        x = self.conv1(x)
        x = self.conv2(x)
        x = channel_shuffle(x, self.num_classes)
        feat = self.conv3(x)
        
        B, C, H, W = x.size()
        x = feat.view(B, self.num_classes, C//self.num_classes, H, W)
        x = x.view(-1, C//self.num_classes, H, W)
       
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = x.view(B, -1)
        
        return x, feat
        
@DETECTORS.register_module()
class FasterRCNN_TB(TwoStageDetector):
    """Implementation of `Faster R-CNN <https://arxiv.org/abs/1506.01497>`_"""

    def __init__(self,
                 backbone,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 pretrained=None):
        super(FasterRCNN_TB, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained)
        
        self.with_attr = True
        if self.with_attr:
            self.attr_head = AttrHead()
            self.attr_loss = CrossEntropyLoss(use_sigmoid=True)
            self.attr_fusion = AttrFusion()
    
    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        
        if self.with_attr:
            attr_x, attr_feat = self.attr_head(x[-1])
            
        if self.with_neck:
            x = self.neck(x)
            
            if self.with_attr:
                x = self.attr_fusion(x, attr_feat)
            
            
        return x, attr_x if self.with_attr else x
    
    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_attrs,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.

            gt_labels (list[Tensor]): class indices corresponding to each box
            
            gt_attrs (list[Tensor]): attribute class indices corresponding to each image

            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

            proposals : override rpn proposals with custom proposals. Use when
                `with_rpn` is False.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        
        losses = dict()
        
        idx, idx_attr = [], []
        for i, gt_attr in enumerate(gt_attrs):
            if gt_attr.sum() >= 0:
                idx_attr.append(i)
            else:
                idx.append(i)
        
        #if len(idx_attr) > 0:
        #    import pdb
        #    pdb.set_trace()
        
        if self.with_attr:
            x, attr_x = self.extract_feat(img)
            gt_attrs = torch.vstack(gt_attrs)
            
            attr_x, gt_attrs = attr_x[idx_attr], gt_attrs[idx_attr]
            x = [t[idx] for t in x]
            img_metas = [img_metas[i] for i in idx]
            gt_bboxes = [gt_bboxes[i] for i in idx]
            gt_labels = [gt_labels[i] for i in idx]
            
            if len(idx_attr) > 0:
                attr_loss = self.attr_loss(attr_x, gt_attrs)
                losses.update({'attr_loss':attr_loss})
        else:
            x = self.extract_feat(img)
        
        if len(idx) == 0:
            return losses
        
        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                x,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg)
            losses.update(rpn_losses)
        else:
            proposal_list = proposals

        roi_losses = self.roi_head.forward_train(x, img_metas, proposal_list,
                                                 gt_bboxes, gt_labels,
                                                 gt_bboxes_ignore, gt_masks,
                                                 **kwargs)
        losses.update(roi_losses)

        return losses
    
    async def async_simple_test(self,
                                img,
                                img_meta,
                                proposals=None,
                                rescale=False):
        """Async test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'
        
        if self.with_attr:
            x, _ = self.extract_feat(img)
        else:
            x = self.extract_feat(img)

        if proposals is None:
            proposal_list = await self.rpn_head.async_simple_test_rpn(
                x, img_meta)
        else:
            proposal_list = proposals

        return await self.roi_head.async_simple_test(
            x, proposal_list, img_meta, rescale=rescale)

    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'

        if self.with_attr:
            x, _ = self.extract_feat(img)
        else:
            x = self.extract_feat(img)

        # get origin input shape to onnx dynamic input shape
        if torch.onnx.is_in_onnx_export():
            img_shape = torch._shape_as_tensor(img)[2:]
            img_metas[0]['img_shape_for_onnx'] = img_shape

        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
        else:
            proposal_list = proposals

        return self.roi_head.simple_test(
            x, proposal_list, img_metas, rescale=rescale)

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        if self.with_attr:
            x, _ = self.extract_feat(img)
        else:
            x = self.extract_feat(img)
        proposal_list = self.rpn_head.aug_test_rpn(x, img_metas)
        return self.roi_head.aug_test(
            x, proposal_list, img_metas, rescale=rescale)
