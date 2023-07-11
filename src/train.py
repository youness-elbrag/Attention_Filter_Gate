from model.Attention_filter_modelimport  import Segementer
from model.loss import BCEDiceLoss
from model.metrics import Scores
from model.meaure_training import ThroughputCallback , CPUUsageCallback
from mode.grad_cam import benchmark
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.warnings import PossibleUserWarning
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import Dataset, DataLoader
import imgaug.augmenters as aai
import torch.optim as optim 
from utils import *
from config import *
from matplotlib.animation import FuncAnimation
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt

class LeftSegementer(pl.LightningModule):
    def __init__(self ,config,output_path, *args, **kwargs):
        super(LeftSegementer,self).__init__()
        warnings.filterwarnings("ignore", category=PossibleUserWarning)
        self.save_hyperparameters()
        self.config = config
        self.config.update(config)
        self.model = Segementer(self.config)  ## attentionGate or Classical UNe)
        
        self.target_layers = [self.model.output.out]
        self.category = self.config["category"]
        self.output_path = output_path

        self.loss = BCEDiceLoss()  
        self.monitor_cam = self.config["monitor_cam"]
        self.Score = Scores()
        self.Optimizer = optim.Adam(self.model.parameters(), lr=self.config["lr"])
        self.list_img_val = {
                         "mri":[] ,
                         "cam":[] ,
                         "mask":[],
                         "pred":[] 
        }
        self.loss_v = []      
    
        
    def forward(self,data):
        return self.model(data)
    
    def training_step(self,batch,batch_idx):
        mri , mask = batch 
        mask = mask.float()
        pred = self(mri)
        loss_ = self.loss(pred , mask)        
        self.Score.update(pred.detach().cpu(),
                          mask.detach().cpu()
                         )
        
        dice, iou = self.Score.get_metrics()
        self.log_dict({"Train Loss": loss_,
                       "Train dice": dice, 
                       "Train iou": iou,
                       },
                        on_epoch=True, 
                        prog_bar=True, 
                        logger=True,
                        sync_dist=True)

        return loss_
    
    def validation_step(self,batch,batch_idx):
        mri , mask = batch 
        mask = mask.float()
        pred = self(mri)
        loss_ = self.loss(pred , mask)
        self.Score.update(pred.detach().cpu(),
                          mask.detach().cpu()
                         )
        
        dice, iou = self.Score.get_metrics()
        self.log_dict({"Val Loss": loss_,
                       "Val dice": dice,
                       "Val iou": iou,
                       },
                        on_epoch=True, 
                        prog_bar=True, 
                        logger=True,
                        sync_dist=True,
                       )
        if batch_idx % 100 == 0:
            Score = [dice, iou  , loss_]
            with torch.enable_grad():
                cam = self.benchmark_monitor(self.model,Score,mri ,
                                                 mask ,
                                                 self.target_layers,
                                                 category=self.category,
                                                 method_use=self.monitor_cam ,
                                                 eigen_smooth=False, 
                                                 aug_smooth=False)
            self.list_img_val["cam"].append(cam)
            self.list_img_val["mri"].append(mri[0][0].detach().cpu().numpy())
            self.list_img_val["mask"].append(mask[0][0].detach().cpu().numpy())
            self.list_img_val["pred"].append(pred[0][0].detach().cpu().numpy())
            self.loss_v.append(loss_)


        return loss_ 
    
    def benchmark_monitor(self ,model , Score:list ,image , mask,target_layers,category:str , method_use:str,eigen_smooth=False, aug_smooth=False):
        methods = [("GradCAM++", GradCAMPlusPlus(model=model, target_layers=target_layers, use_cuda=True)),
                   ("AblationCAM", AblationCAM(model=model, target_layers=target_layers, use_cuda=True))
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
    
    def log_image(self,mri,cam,pred,mask,start_frame,end_frame,name):
        pred = pred > 0.5
        fig, axs = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)
        def update(i):
            index = i + start_frame
            axs[0].clear()
            axs[0].imshow(cam[index,:,:,:],cmap='bone')
            axs[0].imshow(np.ma.masked_where(mask[index,:, :]==0, mask[index,:, :]), cmap='cool_r', alpha=0.5)
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

        ani.save(self.output_path+'image_label_overlay_over_slice_{}.gif'.format(name), writer='ffmpeg')

        plt.close(fig)
        
        self.logger.experiment.add_figure(name, fig,self.global_step)

    def stack_img(self , list_img , batch_idx):
        if batch_idx is not None:
            return np.stack(list_img)
                            
        return f"Out of range batch_idx"
        
    def configure_optimizers(self):
        return [self.Optimizer]

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--Config", type=str, required=True)
    parser.add_argument("--Epochs", type=int, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--output", type=str, required=True)




    args = parser.parse_args()
    if args.device is None:
        args.device = "gpu" if torch.cuda.is_available() else "cpu"
    return args

if __name__== "__main__":
    Config = Intializer()
    Config = write_config_to_yaml(Config)
    args = parse_args()

    # Training parameter
    config=read_config_from_yaml(args.Config)

    # Hypyer-Parameters Training 
    num_workers = config["num_workers"]
    batch_size = args.batch_size
    Epoch = args.Epoch
    output_path = args.output
    train_path = config["train_path"]
    val_path = config["val_path"]

    ## Loadimg Dataloader 
    aug = aai.Sequential([
            aai.Affine(scale=(0.85,1.35),
                    rotate=(-45,45)),
            aai.ElasticTransformation()
        ])
    train_set =Left_DataCostume(train_path,aug)
    val_set =Left_DataCostume(val_path,None)
    Training_Loader = DataLoader(train_set,batch_size=batch_size,num_workers=num_workers, shuffle=True,pin_memory=True,persistent_workers=True)
    Validation_Loader = DataLoader(val_set,batch_size=batch_size,num_workers=num_workers, shuffle=False,pin_memory=True,persistent_workers=True)
    crit_loss = BCEDiceLoss()
    Segement_model = LeftSegementer(Config,output_path)
     ### instanace the model from the class 
    throughput = ThroughputCallback(what = "batches", output=output_path)
    CPUUsage = CPUUsageCallback(what = "epochs",output=output_path)
    Check_Point_Callbacks = ModelCheckpoint(
    monitor="Val dice", 
    save_top_k=12,
    mode="max")
    Trainer = pl.Trainer(accelerator="gpu",devices=1,logger=TensorBoardLogger(save_dir= "./Wieghts/logs"), log_every_n_steps=1,
                     callbacks=[Check_Point_Callbacks,CPUUsage,throughput],                    
                     max_epochs=Epoch,fast_dev_run=False)
    Trainer.fit(Segement_model , Training_Loader,Validation_Loader)
    batch_idx = 50
    cam_val ,mri_val , mask_val , pred_val = Segement_model.list_img_val["cam"],Segement_model.list_img_val["mri"], Segement_model.list_img_val["mask"], Segement_model.list_img_val["pred"]
    cam_val ,img_val , mask_val , pred_val = Segement_model.stack_img(cam_val,batch_idx),Segement_model.stack_img(mri_val,batch_idx) , Segement_model.stack_img(mask_val,batch_idx) , Segement_model.stack_img(pred_val,batch_idx)
    frame = cam_val.shape[0]


    Segement_model.log_image(img_val,cam_val,pred_val,mask_val,start_frame=0,end_frame=(frame - 0),name="val")

