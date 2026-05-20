from .detector3d_template import Detector3DTemplate


class SECONDNet(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()
        self.forward_ret_dict = {}

    def forward(self, batch_dict):
        batch_dict['global_step'] = int(self.global_step.item())
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)
        self.forward_ret_dict = batch_dict

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

        loss_rpn, tb_dict = self.dense_head.get_loss()
        tb_dict = {
            'loss_rpn': loss_rpn.item(),
            **tb_dict
        }
        tb_dict.update(self.forward_ret_dict.get('ground_guided_tb_dict', {}))
        tb_dict.update(self.forward_ret_dict.get('ground_context_tb_dict', {}))
        loss = loss_rpn

        if hasattr(self, 'map_to_bev_module') and hasattr(self.map_to_bev_module, 'get_loss'):
            loss_ground_defect, ground_defect_tb = self.map_to_bev_module.get_loss()
            if loss_ground_defect is not None:
                loss = loss + loss_ground_defect
                tb_dict.update(ground_defect_tb)
                tb_dict['loss_total'] = loss.item()

        return loss, tb_dict, disp_dict
