from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from PIL import Image
import cv2
from pytorch_grad_cam import GradCAMPlusPlus, AblationCAM
from pytorch_grad_cam import GradCAM
from matplotlib.animation import FuncAnimation
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plts

def class_to_idx(class_name:str):
    sem_classes = ['atruim','No atruim ']
    sem_class_to_idx = {cls: idx for (idx, cls) in enumerate(sem_classes)}
    category = sem_class_to_idx[class_name]
    return category

def imgCategory(mask,category):
    mask_trg= mask[0, :, :, :].argmax(axis=0).detach().cpu().numpy()
    mask_uint8 = 255 * np.uint8(mask_trg == category)
    mask = np.float32(mask_trg == category)
    return mask

class SegmentationModelOutputWrapper(torch.nn.Module):
    def __init__(self, model): 
        super(SegmentationModelOutputWrapper, self).__init__()
        self.model = model
        
    def forward(self, x):
        return self.model(x)
    
    
class SemanticSegmentationTarget:
    def __init__(self, category, mask):
        self.category = category
        self.mask = torch.from_numpy(mask)
        if torch.cuda.is_available():
            self.mask = self.mask.cuda()
        
    def __call__(self, model_output):
        return (model_output[self.category, :, : ] * self.mask).sum()

def gray2rgb(img):
    rgb_img = img[0].detach().cpu().numpy()
    backtorgb = cv2.cvtColor(rgb_img[0],cv2.COLOR_GRAY2RGB)
    return backtorgb

def visualize_score(visualization, score, name):
    visualization = cv2.putText(visualization, name, (10, 20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, cv2.LINE_AA)
    visualization = cv2.putText(visualization, "(Least first - Most first)/2", (10, 40), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,255), 1, cv2.LINE_AA)
    visualization = cv2.putText(visualization, f"Dice metric: {score[0]:.3f}", (10, 55), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, cv2.LINE_AA)    
    visualization = cv2.putText(visualization, f"Iou metric: {score[1]:.3f}", (10, 70), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, cv2.LINE_AA) 
    visualization = cv2.putText(visualization, f"Loss Dice :{score[2]:.3f}", (10, 85), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, cv2.LINE_AA)    
    return visualization

def benchmark(model ,image , mask , target_layers,category:str , method_use:str ,eigen_smooth=False, aug_smooth=False):
    methods = [("GradCAM++", GradCAMPlusPlus(model=model, target_layers=target_layers, use_cuda=True)),
               ("AblationCAM", AblationCAM(model=model, target_layers=target_layers, use_cuda=True))
              ]
    method = [tuple([(method_use,cam_use)]) for k , cam_use  in methods if method_use in k][0]

    category = class_to_idx(category)
    mask_float = imgCategory(mask , category)
    targets = [SemanticSegmentationTarget(category, mask_float)]
    
    visualizations = []
    for name, cam_method in method:
        with cam_method:
            attributions = cam_method(input_tensor=image, 
                                      targets=targets, eigen_smooth=eigen_smooth, aug_smooth=aug_smooth)
        attribution = attributions[0, :]  
        prediction = model(image)
        score = Scores()
        score.update(prediction.detach().cpu(),mask.detach().cpu())
        dice , iou , hausdorff = score.get_metrics()
        score = [dice , iou , hausdorff]
        rgb_img = gray2rgb(image)
        visualization = show_cam_on_image(rgb_img, attribution, use_rgb=True)
        visualization = visualize_score(visualization, score, name)
        visualizations.append(visualization)
        
        output = Image.fromarray(np.hstack(visualizations))
        output.save("img1.png")
    
    return visualization


