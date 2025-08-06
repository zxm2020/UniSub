import os
import torch
import torch.nn.functional as F
from torch.nn import CosineSimilarity
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.utils import save_image
from torch.utils.data import Subset
import random

from RestNet import resnet50
from VGG import vgg16
from DenseNet import DenseNet121
from filterDataset import CustomImageDataset
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
from tsrd_dataset import get_datasets

os.environ["TOKENIZERS_PARALLELISM"] = "false"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

label_map = {
    0: "Speed limit 5 km/h",
    1: "Speed limit 15 km/h",
    2: "Speed limit 30 km/h",
    3: "Speed limit 40 km/h",
    4: "Speed limit 50 km/h",
    5: "Speed limit 60 km/h",
    6: "Speed limit 70 km/h",
    7: "Speed limit 80 km/h",
    8: "No left turn or straight ahead",
    9: "No right turn or straight ahead",
    10: "No straight ahead",
    11: "No left turn",
    12: "No left or right turn",
    13: "No right turn",
    14: "No overtaking",
    15: "No U-turn",
    16: "No cars allowed",
    17: "No honking",
    18: "End of 40 km/h limit",
    19: "End of 50 km/h limit",
    20: "Straight or right turn only",
    21: "Straight ahead only",
    22: "Left turn only",
    23: "Left or right turn only",
    24: "Right turn only",
    25: "Keep left",
    26: "Keep right",
    27: "Roundabout",
    28: "Cars only",
    29: "Honking allowed",
    30: "Bicycle lane",
    31: "U-turn only",
    32: "Obstacle ahead",
    33: "Traffic signals ahead",
    34: "Slow down",
    35: "Pedestrian crossing ahead",
    36: "Bicycles ahead",
    37: "School zone ahead",
    38: "Sharp left turn ahead",
    39: "Sharp right turn ahead",
    40: "Steep left descent ahead",
    41: "Steep right descent ahead",
    42: "Slow down",
    43: "T-junction ahead (right)",
    44: "T-junction ahead (left)",
    45: "Village ahead",
    46: "Sharp turn ahead",
    47: "Railway crossing ahead",
    48: "Construction ahead",
    49: "Hairpin turn ahead",
    50: "Railroad crossing",
    51: "Beware of rear-end collisions",
    52: "Stop",
    53: "No entry",
    54: "No parking",
    55: "Do not enter",
    56: "Yield",
    57: "Stop for inspection"
}

def load_model_small(name, device):
    from RestNet import resnet50
    if name == 'ResNet':
        model = resnet50(num_classes=58).to(device)
        model.load_state_dict(torch.load("tsrd/resnet_tsrd.pth", weights_only=True))
    elif modelName == 'VGG':
        model = vgg16(num_classes=58).to(device)
        model.load_state_dict(torch.load("tsrd/vgg_tsrd.pth", weights_only=True))
    elif modelName == 'DenseNet':
        model = DenseNet121(num_classes=58).to(device)
        model.load_state_dict(torch.load("tsrd/densenet_tsrd.pth", weights_only=True))
    elif modelName == 'ViT':
        import sys
        sys.path.append('vision_transformer_pytorch-main')
        from models.vision_transformer import VisionTransformer
        config = {
            'img_size': [224, 224],
            'num_classes': 58,
            'patch_size': [16, 16],
            'hidden_size': 768,
            'mlp_dim': 3072,
            'num_heads': 12,
            'num_layers': 12,
            'dropout_rate': 0.1,
            'attention_dropout_rate': 0.0
        }
        model = VisionTransformer(config).to(device)
        model.load_state_dict(torch.load("tsrd/vit_tsrd.pth", map_location=device,
                       weights_only=True))
    else:
        print('no model')
        exit(0)
    model.eval()
    return model

def modelloader(device):
    model_id = "model/CLIP-vit-large-patch32"
    model = CLIPModel.from_pretrained(model_id).to(device)
    processor = CLIPProcessor.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        tokenizer_file=os.path.join(model_id, "tokenizer.json"),
        trust_remote_code=True
    )
    return model, processor, tokenizer

def build_text_features(model, device):
    prompts = [f"This is a photo showing a {label_map[i]} traffic sign" for i in range(58)]
    processor = CLIPProcessor.from_pretrained("/home/Newdisk2/zhaozhuo/Code/LLM/model/CLIP-vit-large-patch32")
    inputs = processor(text=prompts, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        text_features = model.get_text_features(**inputs)
        text_features = F.normalize(text_features, dim=-1)
    return text_features, prompts

def dataloader():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.3403, 0.3121, 0.3214),
                             (0.2724, 0.2608, 0.2669))])
    trainset, testset = get_datasets("/home/Newdisk2/zhaozhuo/Code/zxm/TSRD", transform)

    indices = random.sample(range(len(trainset)), 1000)
    sampled_dataset = Subset(trainset, indices)

    trainloader = torch.utils.data.DataLoader(sampled_dataset, batch_size=1, shuffle=True, num_workers=4)

    return trainloader


def denormalize(tensor):
    mean = torch.tensor([0.3403, 0.3121, 0.3214]).view(3, 1, 1).to(tensor.device)
    std = torch.tensor([0.2724, 0.2608, 0.2669]).view(3, 1, 1).to(tensor.device)
    return tensor * std + mean

def generate_clip_attack_until_success(modelName_small, model, text_features, prompts, dataloader, device, epsilon=0.03, step_size=0.005, max_steps=30):
    model.eval()
    image_count = 0
    for data, label, fname in tqdm(dataloader, desc="对抗攻击中"):
        data, label = data.to(device), label.to(device)
        img_name = fname[0] if isinstance(fname, (list, tuple)) else fname

        loss_max = 0
        pes_index = 0
        next_step = False

        delta_count = 0
        delta_feature = []

        with torch.no_grad():
            orig_feature = model.get_image_features(data)
            orig_feature = F.normalize(orig_feature, dim=-1)

        for _ in range(100):
            delta = 2*torch.randn_like(data).to(device)
            delta.requires_grad_()

            for lll in range(max_steps):
                adv_img = torch.clamp(data + delta, 0, 1)

                norm = transforms.Normalize((0.3403, 0.3121, 0.3214), (0.2724, 0.2608, 0.2669))
                adv_img_res = norm(adv_img.squeeze(0)).unsqueeze(0)

                adv_features = model.get_image_features(adv_img)
                adv_features = F.normalize(adv_features, dim=-1)

                # loss
                sim = CosineSimilarity(dim=-1)(orig_feature, adv_features)
                loss_clip = -sim
                res_model = load_model_small(modelName_small, device)
                res_out = res_model(adv_img_res)
                loss_resnet = res_out[0][label] * 0.01
                loss = loss_clip + loss_resnet

                model.zero_grad()
                if delta.grad is not None:
                    delta.grad.zero_()
                loss.backward()


                grad = delta.grad.detach()
                delta.data = delta + step_size * grad.sign()
                # === CLIP 输出判断 ===
                with torch.no_grad():
                    logits = adv_features @ text_features.T
                    clip_pred = logits.argmax(dim=-1).item()
                    res_pred = res_out.argmax(dim=-1).item()
                    if clip_pred != label.item() and res_pred != label.item():
                        save_path = f"filter_perturbation/{label.item()}"
                        os.makedirs(save_path, exist_ok=True)
                        save_name = os.path.join(save_path, f"{delta_count}_{img_name}")

                        if loss_max <= loss:
                            loss_max = loss
                            pes_index = delta_count

                        # 保存扰动图像
                        perturb = (adv_img - data).squeeze(0).detach().cpu()
                        perturb_vis = (perturb - perturb.min()) / (perturb.max() - perturb.min() + 1e-8)
                        perturb_pil = TF.to_pil_image(perturb_vis)
                        perturb_pil.save(save_name)

                        perturb_fla = perturb.flatten()
                        delta_feature.append(perturb_fla)

                        print(f"成功攻击样本：original={label.item()} | CLIP={clip_pred} | {modelName_small}={res_pred}")

                        delta_count += 1
                if delta_count == 10:
                    next_step = True
                    break
            if next_step:
                break

        if len(delta_feature) > 0 and next_step == True:
            before_pca = torch.stack(delta_feature, dim=0)  # shape: [10, D]
            final_persu = before_pca[pes_index]

            # centered
            delta_mean = before_pca.mean(dim=0, keepdim=True)
            delta_centered = before_pca - delta_mean

            N, D = delta_centered.shape
            r = D // 2

            U, S, Vh = torch.linalg.svd(delta_centered, full_matrices=False)
            U_r = Vh[:r].T

            z = (final_persu - delta_mean.squeeze(0))  @ U_r
            print("begin attack.......")
            optimizer = torch.optim.Adam([z], lr=step_size)
            z = z.clone().detach().requires_grad_()
            for t in tqdm(range(max_steps)):
                delta_opt = U_r @ z + delta_mean.squeeze(0)
                delta_opt = delta_opt.to(device)
                delta_img = delta_opt.view_as(data) # 还原图像

                adv_img = torch.clamp(data + delta_img, 0, 1)
                norm = transforms.Normalize((0.3403, 0.3121, 0.3214), (0.2724, 0.2608, 0.2669))
                adv_img_res = norm(adv_img.squeeze(0)).unsqueeze(0)
                adv_features = model.get_image_features(adv_img)
                adv_features = F.normalize(adv_features, dim=-1)

                # loss
                sim = CosineSimilarity(dim=-1)(orig_feature, adv_features)
                loss_clip = -sim
                res_model = load_model_small(modelName_small, device)
                res_out = res_model(adv_img_res)
                loss_resnet = res_out[0][label] * 0.01
                reg_loss = torch.norm(delta_opt, p=2)
                lambda_reg = 0.05
                loss = loss_clip + loss_resnet + lambda_reg * reg_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    logits = adv_features @ text_features.T
                    clip_pred = logits.argmax(dim=-1).item()
                    res_pred = res_out.argmax(dim=-1).item()
                    if t == 15:
                        print(f"子空间优化成功 z_step={t+1}, label = {label.item()}, CLIP={clip_pred}, {modelName_small}={res_pred}")
                        save_adv_path = f"/home/Newdisk2/zhaozhuo/Code/zxm/{modelName_small}_CLIP/final_success_adv/{label.item()}"
                        os.makedirs(save_adv_path, exist_ok=True)
                        adv_save_name = os.path.join(save_adv_path, f"adv_{img_name}")
                        adv_pil = TF.to_pil_image(adv_img.squeeze(0).cpu())
                        adv_pil.save(adv_save_name)
                        print(f"已保存{image_count}张对抗图像: {adv_save_name}")
                        image_count += 1
                        break
        else:
            continue

if __name__ == '__main__':
    model, processor, tokenizer = modelloader(device)
    text_features, prompts = build_text_features(model, device)
    loader = dataloader()
    modelName = 'ResNet'
    generate_clip_attack_until_success(
        modelName_small = modelName,
        model=model,
        text_features=text_features,
        prompts=prompts,
        dataloader=loader,
        device=device,
        epsilon=0.03,
        step_size=0.005,
        max_steps=100
    )
