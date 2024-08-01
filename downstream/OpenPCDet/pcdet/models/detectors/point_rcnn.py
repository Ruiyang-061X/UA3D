from .detector3d_template import Detector3DTemplate


class PointRCNN(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()
        self.forward_ret_dict = {}

    def forward(self, batch_dict):
        # for cur_module in self.module_list:
        #     batch_dict = cur_module(batch_dict)
        
        # backbone_3d
        batch_dict, batch_dict2 = self.module_list[0](batch_dict)

        # point_head
        batch_dict = self.module_list[1](batch_dict)
        self.forward_ret_dict['1.pointhead'] = self.module_list[1].forward_ret_dict
        # roi_head
        batch_dict = self.module_list[3](batch_dict)
        self.forward_ret_dict['1.roihead'] = self.module_list[3].forward_ret_dict

        # point_head2
        batch_dict2 = self.module_list[2](batch_dict2)
        self.forward_ret_dict['2.pointhead'] = self.module_list[2].forward_ret_dict
        # roi_head2
        batch_dict2 = self.module_list[4](batch_dict2)
        self.forward_ret_dict['2.roihead'] = self.module_list[4].forward_ret_dict

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

    def get_training_loss(self):
        disp_dict = {}
        loss_point, tb_dict = self.point_head.get_loss(forward_ret_dict_full=self.forward_ret_dict)
        loss_rcnn, tb_dict = self.roi_head.get_loss(tb_dict=tb_dict, forward_ret_dict_full=self.forward_ret_dict)

        loss_point2, tb_dict2 = self.point_head2.get_loss(forward_ret_dict_full=self.forward_ret_dict)
        loss_rcnn2, tb_dict2 = self.roi_head2.get_loss(tb_dict=tb_dict2, forward_ret_dict_full=self.forward_ret_dict)

        loss = loss_point + loss_rcnn + loss_point2 + loss_rcnn2
        return loss, tb_dict, disp_dict