import sys
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

sys.path.insert(0, '/workspace/module/image2text/blip')
from models.blip import blip_decoder

class BLIP:
    def __init__(self):
        self.image_size = 384
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = blip_decoder(pretrained="/workspace/module/weights/model_base_capfilt_large.pth", image_size=self.image_size, vit='base', med_config="/workspace/module/image2text/blip/configs/med_config.json")
        self.model.eval()
        self.model = self.model.to(self.device)

    def load_image(self, image_path):
        raw_image = Image.open(image_path).convert('RGB')

        transform = transforms.Compose([
            transforms.Resize((self.image_size,self.image_size),interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
            ])
        image = transform(raw_image).unsqueeze(0).to(self.device)
        return image

    def generate_caption(self, image_path):
        image = self.load_image(image_path)
        with torch.no_grad():
            captions = self.model.generate(image, sample=False, num_beams=3, max_length=20, min_length=5)
            return captions