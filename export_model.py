import torch

if __name__ == '__main__':
    source_path = "root/model/save/fashion_mnist/V2_iterative_augmenting/DS_fashion_mnist/MT_conv4_dlgn_n16_small_ET_GENERATE_ALL_FINAL_TEMPLATE_IMAGES/_COLL_OV_train/SEG_GT/TMP_COLL_BS_1/TMP_LOSS_TP_TEMP_LOSS/TMP_INIT_zero_init_image/_torch_seed_2022_c_thres_0.73/aug_conv4_dlgn_iter_1_dir.pt"
    target_path = "root/temp/model.pt"
    custom_temp_model = torch.load(source_path)
    torch.save(custom_temp_model.state_dict(), target_path)
