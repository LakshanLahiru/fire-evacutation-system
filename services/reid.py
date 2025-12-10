# services/reid.py
import torch
import torchvision.transforms as T
import numpy as np
from typing import Optional
import torchreid

class ReIDExtractor:
    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # Build model (osnet_x1_0) and load ImageNet pretrained weights shipped by torchreid
        self.model = torchreid.models.build_model(name='osnet_x1_0', num_classes=1000, loss='softmax')
        # try to download the pretrained weight (torchreid helper)
        try:
            weight_path = torchreid.utils.download_url(
                'https://github.com/KaiyangZhou/deep-person-reid/releases/download/v1.0/osnet_x1_0_imagenet.pth',
                model_dir='./model_data', model_name='osnet_x1_0_imagenet.pth'
            )
            torchreid.utils.load_pretrained_weights(self.model, weight_path)
        except Exception as e:
            print("Warning: failed to load pretrained weights:", e)
        self.model.eval().to(self.device)

        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((256, 128)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225])
        ])

    def extract(self, img_bgr) -> Optional[np.ndarray]:
        """
        img_bgr: cropped person (BGR from OpenCV)
        returns L2-normalized numpy vec (C,) or None on failure
        """
        try:
            img_rgb = img_bgr[..., ::-1]  # BGR -> RGB
            x = self.transform(img_rgb).unsqueeze(0).to(self.device)
            with torch.no_grad():
                feat = self.model(x)  # shape (1, C)
            feat = feat.cpu().numpy().reshape(-1)
            # L2 normalize
            n = np.linalg.norm(feat) + 1e-8
            feat = feat / n
            return feat
        except Exception as e:
            print("ReID extract error:", e)
            return None
