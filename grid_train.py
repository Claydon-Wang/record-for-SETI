import os

# os.system("python train.py --cfg ./configs/seti_320_r0_1.yaml DATA.TRAIN.AUG [['Aresize',[320,320,2]],['AHorizontalFlip',[0.5]],['AVerticalFlip',[0.5]],['AShiftScaleRotate',[0.2,0.2,0]],['AToTensorV2',[]]]")
os.system("python train.py --cfg ./configs/seti_320_r0_1.yaml TRAINOR.LOSSNAME BCEFocal_loss TRAINOR.ALPHA_LOSS 0.65 TRAINOR.GAMMA 1.0")
# os.system("python train.py --cfg ./configs/seti_320_r0_1.yaml TRAINOR.LOSSNAME BCEFocal_loss TRAINOR.ALPHA_LOSS 0.35 TRAINOR.GAMMA 1.0")
# os.system("python train.py --cfg ./configs/seti_320_r0_1.yaml TRAINOR.LOSSNAME BCEFocal_loss TRAINOR.ALPHA_LOSS 0.5 TRAINOR.GAMMA 1.0")



# param_list = [
 
#     ['tf_efficientnet_b0_ns', 320, True, 0.5, 20, 2.0, 7, 1, "adamw", 0.001],
# #     ['tf_efficientnet_b0_ns', 320, True, 0.2, 20, 2.0, 7, 6, "adamw", 0.001],
# #     ['tf_efficientnet_b0_ns', 320, True, 0.5, 20, 2.0, 7, 3, "adamw", 0.001],
    

#   ]


# for param in param_list:
#     os.system(f"python train.py --cfg ./configs/seti_{param[1]}_r{param[7]}.yaml MODEL.NAME {param[0]} MODEL.SELF_HEAD {param[2]} MODEL.DROPOUT {param[3]} OPTIM.EPOCHS {param[4]} TRAINOR.ALPHA {param[5]} RNG_SEED {param[6]} OPTIM.OPT {param[8]} OPTIM.LR {param[9]}")