'''import torch
import gc
import os
import time
import numpy as np
import shutil
from PIL import Image, ImageOps
import requests
from io import BytesIO
from tqdm import tqdm
from time import sleep
import random
from controlnet_aux.util import HWC3
from cv_utils import resize_image

from diffusers import (
   ControlNetModel, DiffusionPipeline, StableDiffusionControlNetPipeline, UniPCMultistepScheduler
)

#ControlNetModel.from_pretrained('lllyasviel/control_v11f1p_sd15_depth')
#controlnet = ControlNetModel.from_pretrained('runwayml/stable-diffusion-v1-5',torch_dtype=torch.float16)
'''

'''currentmodel = "runwayml/stable-diffusion-v1-5"

_seed = random.randint(0, 2147483647)

generator = torch.Generator('cuda').manual_seed(_seed)

controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11f1p_sd15_depth",torch_dtype=torch.float16)

pipe = StableDiffusionControlNetPipeline.from_pretrained("runwayml/stable-diffusion-v1-5",controlnet=controlnet,torch_dtype=torch.float16)
'''


import cv2
import torch
import numpy as np
from PIL import Image
import random
from controlnet_aux.util import HWC3
from transformers import pipeline
from diffusers.utils import load_image
from diffusers import (
   ControlNetModel, DiffusionPipeline, StableDiffusionControlNetPipeline, UniPCMultistepScheduler, PNDMScheduler,LMSDiscreteScheduler,DDIMScheduler,EulerDiscreteScheduler,EulerAncestralDiscreteScheduler,DPMSolverMultistepScheduler
)

checkpoint = "lllyasviel/control_v11p_sd15_depth"

#image = load_image("https://huggingface.co/lllyasviel/control_v11p_sd15_depth/resolve/main/images/input.png")

image = Image.open("source_image.png")
prompt = "Stormtrooper's lecture in beautiful lecture hall"

depth_estimator = pipeline('depth-estimation')
image = depth_estimator(image)['depth']
image = np.array(image)
image = image[:, :, None]
image = np.concatenate([image, image, image], axis=2)
control_image = Image.fromarray(image)

control_image.save("control.png")

controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11f1p_sd15_depth",torch_dtype=torch.float16)
pipe = StableDiffusionControlNetPipeline.from_pretrained("runwayml/stable-diffusion-v1-5",controlnet=controlnet,torch_dtype=torch.float16)
pipe.to("cuda")
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_xformers_memory_efficient_attention()
pipe.enable_model_cpu_offload()

_seed = random.randint(0, 2147483647)
generator = torch.Generator('cuda').manual_seed(_seed)

_image = pipe(prompt, num_inference_steps=30, generator=generator, image=control_image).images[0]
_image.save('image_out.png')

#controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_depth", torch_dtype=torch.float16)
#pipe = StableDiffusionControlNetPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16)


'''pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()

generator = torch.manual_seed(0)
image = pipe(prompt, num_inference_steps=30, generator=generator, image=control_image).images[0]

image.save('images/image_out.png')'''