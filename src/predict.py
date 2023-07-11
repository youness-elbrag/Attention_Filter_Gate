
import glob, os
from utils import * 
from train import LeftSegementer
import torch
from matplotlib.animation import FuncAnimation
from mpl_toolkits.axes_grid1 import make_axes_locatable
import nibabel as nib

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--Config", type=str, required=True)
    parser.add_argument("--sample", type=int, required=True)
    parser.add_argument("--output", type=str, required=True)

   
if __name__=="__main__":
    Config = Intializer()
    Config = write_config_to_yaml(Config)
    args = parse_args()
    config=read_config_from_yaml(args.Config)
    output = args.output
    sample = args.sample

    logs ="./Wieghts/logs"
    path_model = []
    for root, dirs, files in os.walk(logs):
        for file in files:
            if file.endswith(".ckpt"):
                path_model.append(os.path.join(root, file))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_load = LeftSegementer(config)

    PreTrained = model_load.load_from_checkpoint(path_model[-1])
    PreTrained.to(device)
    PreTrained.eval();
    list_img_test = {
                    "mri":[] ,
                    "cam":[] ,
                    "pred":[] 
    }
    # Prediction :
    #sample = "/heart-mri-image-dataset-left-atrial-segmentation/imagesTs/la_001.nii"
    Path_test = sample
    image = nib.load(Path_test)
    ## assert the Voxel Corrd 
    assert nib.aff2axcodes(image.affine) == ('R', 'A', 'S')
    ## get the array values
    image_array = image.get_fdata()
    ## Crop region through 2D shape 
    crop_2d_img = image_array[72:-72,72:-72]
    ## normalize and Standrize Image only 
    std_mri = Standrization(Normalization(crop_2d_img))

    for i in tqdm(range(std_mri.shape[-1])):
        slice_ = std_mri[:,: ,i]
        slice = torch.tensor(slice_).unsqueeze(0).unsqueeze(0).float().to(device)
        
        with torch.no_grad():
            pred = PreTrained(slice)
        Score = [mean_dice , mean_iou , 0.12]
        
        torch.set_grad_enabled(True)
        with torch.enable_grad():
            cam = benchmark_monitor(model=PreTrained,Score=Score,image=slice ,
                                                mask=pred ,
                                                target_layers= PreTrained.target_layers,
                                                category=PreTrained.category,
                                                method_use=PreTrained.monitor_cam ,
                                                eigen_smooth=False, 
                                                aug_smooth=False,
                                                cuda=True)

            list_img_test["cam"].append(cam)
            list_img_test["mri"].append(slice_)
            list_img_test["pred"].append(pred[0][0].detach().cpu().numpy())


        cam_test = list_img_test["cam"]
        mri_test  = list_img_test["mri"]
        pred_test =  list_img_test["pred"]

        batch_idx= 1
        cam_test = stack_img(cam_test,batch_idx)
        mri_test  = stack_img(mri_test,batch_idx) 
        pred_test =stack_img(pred_test,batch_idx)

        frame = cam_test.shape[0]

        log_image(mri_test,cam_test,pred_test,start_frame=0,end_frame=(frame - 1),name="Test")

        show_gif( output+'image_label_overlay_over_slice_Prediction_Test.gif', format='png')

