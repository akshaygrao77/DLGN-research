import torch

if __name__ == '__main__':
    source_path = "root/model/save/cifar10/adversarial_training/MT_dlgn__conv4_dlgn_pad_k_1_st1_bn_wo_bias___ET_ADV_TRAINING/ST_2022/fast_adv_attack_type_PGD/adv_type_PGD/EPS_0.06/batch_size_128/eps_stp_size_0.06/adv_steps_80/adv_model_dir.pt"
    target_path = "root/temp/model.pt"
    custom_temp_model = torch.load(source_path)
    torch.save({'state_dict':custom_temp_model.state_dict()}, target_path)
