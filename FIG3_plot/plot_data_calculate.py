import os
from diffusers import DDPMPipeline, DDPMScheduler, UNet2DModel
from diffusers.schedulers.scheduling_ddpm import DDPMSchedulerOutput
import torch
import os
import numpy as np
from scipy.stats import mode
from scipy.interpolate import griddata
import seaborn as sns
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import transforms
import torchvision
from torch.utils.data import DataLoader
from typing import List, Optional, Tuple, Union
from diffusers.pipelines.pipeline_utils import ImagePipelineOutput
from tqdm import tqdm
from loguru import logger
from torch.utils.checkpoint import checkpoint
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group
from torch.distributed import all_gather
from networks.resnet import ResNet18
from networks.WRN import WideResNet
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--local-rank", type=int, default=-1)
parser.add_argument("--dataset", type=str, default='cifar10')
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--num_test", type=int, default=128)
parser.add_argument("--pgd_step", type=int, default=20)
parser.add_argument("--eot_step", type=int, default=10)
parser.add_argument("--denoise_strength", type=float, default=0.1)
parser.add_argument("--inference_step", type=int, default=50)
parser.add_argument("--model_id", type=str, default='../../train/ddpm_ema_cifar10')
# parser.add_argument("--model_id", type=str, default='../../train/ddpm-addt003')
parser.add_argument("--model_type", type=str, default='t7')
parser.add_argument("--fix_type", type=str, default='large')
parser.add_argument("--save_folder", type=str, default=None)
parser.add_argument("--savedir", type=str, default=None)
FLAGS = parser.parse_args()

local_rank = FLAGS.local_rank
init_process_group(backend="nccl")
torch.cuda.set_device(FLAGS.local_rank)
world_size = torch.distributed.get_world_size()
device = torch.device("cuda", local_rank)


Denoise_strength = FLAGS.denoise_strength
Num_inference_steps = FLAGS.inference_step
model_id = FLAGS.model_id
savedir = FLAGS.savedir
batch_size = FLAGS.batch_size
num_test = FLAGS.num_test
pgd_step = FLAGS.pgd_step
eot_step = FLAGS.eot_step


class DDPMPipeline_Img2Img(DDPMPipeline):
    def __call__(
        self,
        sample_image,
        batch_size: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        timesteps_list = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        reverse_noise = None 
    ) -> Union[ImagePipelineOutput, Tuple]:

        image = sample_image
        # set step values
        self.unet.eval()
        for t in timesteps_list:
            # 1. predict noise model_output
            model_output = checkpoint(self.unet, image, t, None, False)[0]
            # 2. compute previous image: x_t -> x_t-1
            image = self.scheduler.step(model_output, t, image, generator=generator, reverse_noise=reverse_noise).prev_sample

        image = (image / 2 + 0.5).clamp(0, 1)
        image1 = image.cpu().permute(0, 2, 3, 1).detach().numpy()
        if output_type == "pil":
            image1 = self.numpy_to_pil(image1)
        return image, image1


class DDPMScheduler_Wrap(DDPMScheduler):
    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: int,
        sample: torch.FloatTensor,
        generator=None,
        return_dict: bool = True,
        reverse_noise = None,
    ) -> Union[DDPMSchedulerOutput, Tuple]:
        t = timestep

        prev_t = self.previous_timestep(t)

        if model_output.shape[1] == sample.shape[1] * 2 and self.variance_type in ["learned", "learned_range"]:
            model_output, predicted_variance = torch.split(model_output, sample.shape[1], dim=1)
        else:
            predicted_variance = None

        # 1. compute alphas, betas
        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1 - current_alpha_t

        # 2. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (15) from https://arxiv.org/pdf/2006.11239.pdf
        if self.config.prediction_type == "epsilon":
            pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
        elif self.config.prediction_type == "sample":
            pred_original_sample = model_output
        elif self.config.prediction_type == "v_prediction":
            pred_original_sample = (alpha_prod_t**0.5) * sample - (beta_prod_t**0.5) * model_output
        else:
            raise ValueError(
                f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample` or"
                " `v_prediction`  for the DDPMScheduler."
            )

        # 3. Clip or threshold "predicted x_0"
        if self.config.thresholding:
            pred_original_sample = self._threshold_sample(pred_original_sample)
        elif self.config.clip_sample:
            pred_original_sample = pred_original_sample.clamp(
                -self.config.clip_sample_range, self.config.clip_sample_range
            )

        # 4. Compute coefficients for pred_original_sample x_0 and current sample x_t
        # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
        pred_original_sample_coeff = (alpha_prod_t_prev ** (0.5) * current_beta_t) / beta_prod_t
        current_sample_coeff = current_alpha_t ** (0.5) * beta_prod_t_prev / beta_prod_t

        # 5. Compute predicted previous sample µ_t
        # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
        pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * sample

        # 6. Add noise
        variance = 0
        if t > 0:
            device = model_output.device
            # variance_noise = randn_tensor(
            #     model_output.shape, generator=generator, device=device, dtype=model_output.dtype
            # )
            variance_noise = reverse_noise[t].to(device)
            if self.variance_type == "fixed_small_log":
                variance = self._get_variance(t, predicted_variance=predicted_variance) * variance_noise
            elif self.variance_type == "learned_range":
                variance = self._get_variance(t, predicted_variance=predicted_variance)
                variance = torch.exp(0.5 * variance) * variance_noise
            else:
                variance = (self._get_variance(t, predicted_variance=predicted_variance) ** 0.5) * variance_noise

        pred_prev_sample = pred_prev_sample + variance

        if not return_dict:
            return (pred_prev_sample,)

        return DDPMSchedulerOutput(prev_sample=pred_prev_sample, pred_original_sample=pred_original_sample)
    
    

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])
if FLAGS.model_type == "t7":
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)
elif FLAGS.model_type == 'cifar100':
    mean = (0.5071, 0.4867, 0.4408)
    std = (0.2675, 0.2565, 0.2761)
elif FLAGS.model_type == "r18" or FLAGS.model_type == "vit":
    mean = (0.0, 0.0, 0.0)
    std = (1.0, 1.0, 1.0)
 
mu = torch.tensor(mean).view(3, 1, 1).to(device)
std1 = torch.tensor(std).view(3, 1, 1).to(device)
ppp = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1).to(device)
upper_limit = ((1 - ppp)/ ppp)
lower_limit = ((0 - ppp)/ ppp)


if FLAGS.model_type == "vit":
    transform_cifar10 = transforms.Compose(
        [transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC)]
    )
else:
    transform_cifar10 = transforms.Compose(
        [transforms.Normalize(mean, std)]
    )
if FLAGS.dataset=='cifar10':
    cifar10_test = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_test)
elif FLAGS.dataset=='cifar100':
    cifar10_test = torchvision.datasets.CIFAR100(root='../data', train=False, download=True, transform=transform_test)
cifar10_test.data = cifar10_test.data[:num_test]
cifar10_test.targets = cifar10_test.targets[:num_test]

sampler = DistributedSampler(cifar10_test, num_replicas=world_size, rank=local_rank)
cifar10_test_loader = DataLoader(
    cifar10_test, shuffle=False, num_workers=5, batch_size=batch_size, sampler=sampler, drop_last=True)

if FLAGS.model_type == "t7":
    states_att = torch.load('../origin.t7', map_location='cpu')  # Temporary t7 setting
    network_clf = states_att['net'].to(device)

network_clf.eval()

if FLAGS.fix_type == 'small':
    noise_schdeuler1 = DDPMScheduler_Wrap(num_train_timesteps=1000)
else:
    noise_schdeuler1 = DDPMScheduler_Wrap(num_train_timesteps=1000, variance_type="fixed_large")

noise_schdeuler1.set_timesteps(num_inference_steps=Num_inference_steps) 
timesteps_list = torch.LongTensor(noise_schdeuler1.timesteps[(round((1-Denoise_strength)*len(noise_schdeuler1.timesteps))-1):])
timesteps = timesteps_list[0]

unet = UNet2DModel.from_pretrained(model_id, subfolder='unet').to(device)
ddpm = DDPMPipeline_Img2Img(unet, noise_schdeuler1).to(device)

epsilon = (8 / 255.) / ppp
alpha = (2 / 255.) / ppp
def clamp1(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)


cls_acc_list = []
pure_cls_acc_list = []
cls_init_acc_list = []

for sample_image, y_val in tqdm(cifar10_test_loader, colour='yellow'):
    sample_image = sample_image.to(device)
    y_val = y_val.to(device)
        
    eot_perturbations = torch.load(f'./{savedir}/eot_perturbations.pth', map_location=device) 
    EOT_Perturbations = torch.unbind(eot_perturbations)
    eot1_perturbations = torch.load(f'./{savedir}/eot1_perturbations.pth', map_location=device) 
    EOT1_Perturbations = torch.unbind(eot1_perturbations)
        
    exact_perturbations = torch.load(f'./{savedir}/exact_perturbations.pth', map_location=device)
    Exact_Perturbations = torch.unbind(exact_perturbations)
    noise_global = torch.load(f'./{savedir}/noise_global_exact.pth', map_location=device)
    reverse_noise = torch.load(f'./{savedir}/reverse_noise_exact.pth', map_location=device)

    #get Direction and Track
    dir1_temp = Exact_Perturbations[-1].flatten(1,3)
    dir1= dir1_temp / torch.norm(dir1_temp, p=2, dim=-1, keepdim=True) / torch.norm(dir1_temp, p=2, dim=-1, keepdim=True)
    dir2 = EOT_Perturbations[-1].flatten(1,3) - Exact_Perturbations[-1].flatten(1,3) * ((torch.nn.functional.cosine_similarity(EOT_Perturbations[-1].flatten(1,3), Exact_Perturbations[-1].flatten(1,3), dim=-1) * torch.norm(EOT_Perturbations[-1].flatten(1,3), p=2, dim=-1) / torch.norm(Exact_Perturbations[-1].flatten(1,3), p=2, dim=-1)))[:, None]
    
    dir2 = dir2 / torch.norm(dir2, p=2, dim=-1, keepdim=True) / torch.norm(dir1_temp, p=2, dim=-1, keepdim=True)
    #print(torch.nn.functional.cosine_similarity(dir1,dir2,dim=-1))
    EOT_Track = [(torch.sum(pos.flatten(1,3) * dir1).item()/batch_size, torch.sum(pos.flatten(1,3) * dir2).item()/batch_size) for pos in EOT_Perturbations]
    EOT1_Track = [(torch.sum(pos.flatten(1,3) * dir1).item()/batch_size, torch.sum(pos.flatten(1,3) * dir2).item()/batch_size) for pos in EOT1_Perturbations]
    Exact_Track = [(torch.sum(pos.flatten(1,3) * dir1).item()/batch_size, torch.sum(pos.flatten(1,3) * dir2).item()/batch_size if torch.sum(pos.flatten(1,3) * dir2).item()/batch_size>=0 else 0.0 ) for pos in Exact_Perturbations]

       
images = []
loss_list = []
pre_list = []

loss_function=nn.CrossEntropyLoss(reduction='none')
range_x=(0,2)
range_y=(0,2)
grid_size=30
rx = np.linspace(*range_x, grid_size)
ry = np.linspace(*range_y, grid_size)
batch_size_iner = 5
adv_direction_acc = ((Exact_Perturbations[-1])/2)
print(adv_direction_acc.shape)
adv_direction = (dir2 * torch.norm(dir1_temp, p=2, dim=-1, keepdim=True) * torch.norm(dir1_temp, p=2, dim=-1, keepdim=True)/2).reshape(-1,3,32,32)
print(adv_direction.shape)
vec_x=adv_direction_acc
vec_y=adv_direction
X, Y = np.meshgrid(rx, ry)
for j in tqdm(ry):
    for i in rx:
        images.append((sample_image/2)+0.5 + i*vec_x + j*vec_y)
        
        if len(images) == batch_size_iner:
            images = torch.stack(images)
            labels = torch.stack([y_val]*batch_size_iner)
            with torch.no_grad():
                noisy_image = noise_schdeuler1.add_noise((images-0.5)*2, noise_global[None,:,:,:,:], timesteps).reshape(batch_size*batch_size_iner,*(images.shape[2:]))
                images_1, images_2 = ddpm(sample_image=noisy_image.to(device), batch_size=noisy_image.shape[0], timesteps_list=timesteps_list, reverse_noise=reverse_noise.repeat(1,images.shape[0],1,1,1))
                outputs = network_clf(transform_cifar10(images_1.to(device)))
            
            _, pres = torch.max(outputs.data, 1)
            
            correct_predictions = (pres == labels.flatten()).reshape(batch_size_iner, -1)
            correct_predictions_count = correct_predictions.sum(dim=1)
            total_predictions_per_batch = correct_predictions.shape[1]
            batch_accuracy = correct_predictions_count.float() / total_predictions_per_batch
            batch_accuracy = batch_accuracy.cpu().numpy()
            pre_list.append(batch_accuracy)
            loss_list.append(torch.mean(loss_function(outputs, labels.flatten()).reshape(-1,batch_size), dim=-1).cpu().numpy())
                
            images = []
if len(images)!=0:
    images = torch.stack(images)
    labels = torch.stack([y_val]*len(images))
    with torch.no_grad():
        remian_batch_size = images.shape[0]
        noisy_image = noise_schdeuler1.add_noise((images-0.5)*2, noise_global[None,:,:,:,:], timesteps).reshape(batch_size*remian_batch_size,*(images.shape[2:]))
        images_1, images_2 = ddpm(sample_image=noisy_image.to(device), batch_size=noisy_image.shape[0], timesteps_list=timesteps_list, reverse_noise=reverse_noise.repeat(1,images.shape[0],1,1,1))
        outputs = network_clf(transform_cifar10(images_1.to(device)))

    _, pres = torch.max(outputs.data, 1)
    correct_predictions = (pres == labels.flatten()).reshape(remian_batch_size, -1)
    correct_predictions_count = correct_predictions.sum(dim=1)
    total_predictions_per_batch = correct_predictions.shape[1]
    batch_accuracy = correct_predictions_count.float() / total_predictions_per_batch
    batch_accuracy = batch_accuracy.cpu().numpy()
    pre_list.append(batch_accuracy)
    loss_list.append(torch.mean(loss_function(outputs, labels.flatten()).reshape(-1,batch_size), dim=-1).cpu().numpy())
pre_list = np.concatenate(pre_list).reshape(len(rx), len(ry))
loss_list = np.concatenate(loss_list).reshape(len(rx), len(ry))


grid_x = X.ravel()
grid_y = Y.ravel()
grid_z = loss_list.ravel()  
EOT_Track_Z = griddata((grid_x, grid_y), grid_z, np.array([(0, 0)] + EOT_Track), method='cubic')
EOT1_Track_Z = griddata((grid_x, grid_y), grid_z, np.array([(0, 0)] + EOT1_Track), method='cubic')
Exact_Track_Z = griddata((grid_x, grid_y), grid_z, np.array([(0, 0)] + Exact_Track), method='cubic')
EOT_Track_3D = []
for i, ((x,y),z) in enumerate(zip(([(0, 0)] + EOT_Track), EOT_Track_Z)):
    EOT_Track_3D.append((x,y,z))
EOT1_Track_3D = []
for i, ((x,y),z) in enumerate(zip(([(0, 0)] + EOT1_Track), EOT1_Track_Z)):
    EOT1_Track_3D.append((x,y,z))
Exact_Track_3D = []
for i, ((x,y),z) in enumerate(zip(([(0, 0)] + Exact_Track), Exact_Track_Z)):
    Exact_Track_3D.append((x,y,z))
    
    
torch.save(loss_list, f'./{savedir}/loss_list.pth')
torch.save(pre_list, f'./{savedir}/pre_list.pth')
torch.save(EOT_Track_3D, f'./{savedir}/eot_track_3d.pth')
torch.save(Exact_Track_3D, f'./{savedir}/exact_track_3d.pth')
torch.save(EOT1_Track_3D, f'./{savedir}/eot1_track_3d.pth')