
import yaml 
import numpy as np 
from src.grad_cam import * 
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from PIL import Image
import cv2
from pytorch_grad_cam import GradCAM, AblationCAM

def write_config_to_yaml(config):
    with open("config.yml", "w") as file:
        yaml.dump(config, file)

def read_config_from_yaml(Path_config):
    with open(Path_config, "r") as file:
        config = yaml.safe_load(file)
    return config

def Loader_Data(path):
    return np.load(path).astype(np.float32)


def benchmark_monitor(model , Score:list ,image , mask,target_layers,category:str , method_use:str,eigen_smooth=False, aug_smooth=False,cuda=False):
    methods = [("GradCAM++", GradCAMPlusPlus(model=model, target_layers=target_layers, use_cuda=cuda)),
               ("AblationCAM", AblationCAM(model=model, target_layers=target_layers, use_cuda=cuda))
              ]
    method = [tuple([(method_use,cam_use)]) for k , cam_use  in methods if method_use in k][0]

    category = class_to_idx(category)
    mask_float = imgCategory(mask , category)
    targets = [SemanticSegmentationTarget(category, mask_float)]
    for name, cam_method in method:
        with cam_method:
            np.random.seed(42)
            attributions = cam_method(input_tensor=image, 
                                      targets=targets, eigen_smooth=eigen_smooth, aug_smooth=aug_smooth)
        attribution = attributions[0, :]  
        rgb_img = gray2rgb(image)
        Cam = show_cam_on_image(rgb_img, attribution, use_rgb=True)
        Cam = visualize_score(Cam, Score, name)

    return Cam
    
def log_image(mri,cam,pred,start_frame,end_frame,name):
    pred = pred > 0.5
    fig, axs = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)
    def update(i):
        index = i + start_frame
        axs[0].clear()
        axs[0].imshow(cam[index,:,:,:],cmap='bone')
        axs[0].imshow(np.ma.masked_where(pred[index,:, :]==0, pred[index,:, :]), cmap='cool_r', alpha=0.5)
        axs[0].set_title('Grouth Truth Images Overlay', fontsize=16, fontweight='bold')
        axs[0].set_xlabel('X-axis', fontsize=14, fontweight='bold')
        axs[0].set_ylabel('Y-axis', fontsize=14, fontweight='bold')
        axs[0].set_xlim([0, mri.shape[1]])
        axs[0].set_ylim([mri.shape[2], 0])

        axs[1].clear()
        axs[1].imshow(mri[index,:,:], cmap='bone')
        axs[1].imshow(np.ma.masked_where(pred[index,:, :]==0, pred[index,:, :]), cmap='cool_r', alpha=0.5)
        axs[1].set_title('Prediction Images Overlay', fontsize=16, fontweight='bold')
        axs[1].set_xlabel('X-axis', fontsize=14, fontweight='bold')
        axs[1].set_ylabel('Y-axis', fontsize=14, fontweight='bold')
        axs[1].set_xlim([0, mri.shape[1]])
        axs[1].set_ylim([mri.shape[2], 0])

        fig.suptitle('Slice:  {:03d}   of  Total {}'.format(i, mri.shape[0]), fontsize=20, fontweight='bold')

        fig.canvas.draw()

    ani = FuncAnimation(fig, update, frames=end_frame, interval=150)

    ani.save('image_label_overlay_over_slice_Prediction_{}.gif'.format(name), writer='ffmpeg')

    plt.close(fig)
        
def stack_img( list_img , batch_idx):
        if batch_idx is not None:
            return np.stack(list_img)
                            
        return f"Out of range batch_idx"

def Normalization(full_volume):
    mean = full_volume.mean()
    std = np.std(full_volume)
    normalized = (full_volume - mean ) / std
    return normalized

def Standrization(normalized_img):
    Standrization = (normalized_img - normalized_img.min()) / (normalized_img.max() - normalized_img.min())
    return Standrization