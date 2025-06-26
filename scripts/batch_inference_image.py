import argparse, os, sys, glob
import cv2
import torch
import numpy as np
from omegaconf import OmegaConf
import pandas as pd
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
import time
from pytorch_lightning import seed_everything
from torch.amp import autocast
from contextlib import contextmanager, nullcontext
import torchvision
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
import torchvision.transforms as transforms
import albumentations as A
import tempfile

from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import AutoFeatureExtractor

from src.utils.alignmengt import crop_faces, calc_alignment_coefficients
from ldm.data.video_swap_dataset import VideoDataset
from torchvision.transforms import Resize

from pretrained.face_parsing.face_parsing_demo import init_faceParsing_pretrained_model, faceParsing_demo

# load safety model
# Note: It's better to move these inside the main function or a setup function 
# if they are not needed globally. For now, keeping original structure.
try:
    safety_model_id = "CompVis/stable-diffusion-safety-checker"
    safety_feature_extractor = AutoFeatureExtractor.from_pretrained(safety_model_id)
    safety_checker = StableDiffusionSafetyChecker.from_pretrained(safety_model_id)
except Exception as e:
    print(f"Could not load safety checker: {e}")
    safety_feature_extractor = None
    safety_checker = None

def get_tensor(normalize=True, toTensor=True):
    transform_list = []
    if toTensor:
        transform_list += [torchvision.transforms.ToTensor()]

    if normalize:
        transform_list += [torchvision.transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
    return torchvision.transforms.Compose(transform_list)

def get_tensor_clip(normalize=True, toTensor=True):
    transform_list = []
    if toTensor:
        transform_list += [torchvision.transforms.ToTensor()]

    if normalize:
        transform_list += [torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                                (0.26862954, 0.26130258, 0.27577711))]
    return torchvision.transforms.Compose(transform_list)

def crop_and_align_face_wrapper(target_files):
    image_size = 1024
    scale = 1.0
    center_sigma = 0
    xy_sigma = 0
    use_fa = False
    
    # print('Aligning images')
    crops, orig_images, quads = crop_faces(image_size, target_files, scale, center_sigma=center_sigma, xy_sigma=xy_sigma, use_fa=use_fa)
    
    if not crops:
        return None, None, None, None

    inv_transforms = [
        calc_alignment_coefficients(quad + 0.5, [[0, 0], [0, image_size], [image_size, image_size], [image_size, 0]])
        for quad in quads
    ]
    
    return crops, orig_images, quads, inv_transforms


def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model

def load_replacement(x):
    try:
        hwc = x.shape
        y = Image.open("assets/rick.jpeg").convert("RGB").resize((hwc[1], hwc[0]))
        y = (np.array(y)/255.0).astype(x.dtype)
        assert y.shape == x.shape
        return y
    except Exception:
        return x

def check_safety(x_image):
    if safety_checker is None or safety_feature_extractor is None:
        return x_image, [False] * x_image.shape[0]
        
    safety_checker_input = safety_feature_extractor(numpy_to_pil(x_image), return_tensors="pt")
    x_checked_image, has_nsfw_concept = safety_checker(images=x_image, clip_input=safety_checker_input.pixel_values)
    assert x_checked_image.shape[0] == len(has_nsfw_concept)
    for i in range(len(has_nsfw_concept)):
        if has_nsfw_concept[i]:
            x_checked_image[i] = load_replacement(x_checked_image[i])
    return x_checked_image, has_nsfw_concept


def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--csv_path",
        type=str,
        required=True,
        help="Path to the input CSV file with 'source' and 'target' columns."
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="batch_results"
    )
    # --- Arguments from one_inference.py ---
    parser.add_argument(
        "--skip_grid",
        action='store_true',
        help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )
    parser.add_argument(
        "--fixed_code",
        action='store_true',
        help="if enabled, uses the same starting code across samples ",
        default=False
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=1,
        help="how many samples to produce for each given prompt. A.k.a. batch size",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=512,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=5.0,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/debug.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="models/REFace/checkpoints/last.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )
    parser.add_argument('--faceParser_name', default='default', type=str, help='face parser name, [ default | segnext] is currently supported.')
    parser.add_argument('--faceParsing_ckpt', type=str, default="Other_dependencies/face_parsing/79999_iter.pth")  
    parser.add_argument('--segnext_config', default='', type=str, help='Path to pre-trained SegNeXt faceParser configuration file.')
    parser.add_argument('--save_vis', action='store_true')
    parser.add_argument('--seg12',default=True, action='store_true')
    parser.add_argument('--disable_safety_checker', action='store_true', help='Disable the NSFW safety checker.')

    opt = parser.parse_args()
    print(opt)
    
    # Disable safety checker if requested
    if opt.disable_safety_checker:
        print("NSFW safety checker is disabled by user flag.")
        global safety_checker, safety_feature_extractor
        safety_checker = None
        safety_feature_extractor = None

    seed_everything(opt.seed)

    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    if opt.plms:
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)

    # Create a timestamped directory for this run's image results
    time_str = time.strftime("%Y%m%d-%H%M%S")
    image_results_dir = os.path.join(opt.outdir, time_str)
    os.makedirs(image_results_dir, exist_ok=True)
    
    faceParsing_model = init_faceParsing_pretrained_model(opt.faceParser_name, opt.faceParsing_ckpt, opt.segnext_config)
    
    try:
        df = pd.read_csv(opt.csv_path)
    except FileNotFoundError:
        print(f"Error: CSV file not found at {opt.csv_path}")
        return

    results_data = []
    
    # batch_size should be 1 for this script's logic
    batch_size = 1

    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing images"):
        start_time_row = time.time()
        
        source_image_path = row['source']
        target_image_path = row['target']
        
        # Determine the number of steps for this specific pair
        steps_for_this_run = opt.ddim_steps  # Default from command line
        if 'steps' in df.columns and pd.notna(row['steps']):
            try:
                steps_for_this_run = int(row['steps'])
            except (ValueError, TypeError):
                # Keep default if value is not a valid integer, maybe log this event
                print(f"\nWarning: Invalid 'steps' value '{row['steps']}' for row {index}. Using default {opt.ddim_steps}.")
                pass

        if not os.path.exists(source_image_path):
            print(f"\nSource image not found: {source_image_path}, skipping row {index}")
            continue
        if not os.path.exists(target_image_path):
            print(f"Target image not found: {target_image_path}, skipping row {index}")
            continue

        with tempfile.TemporaryDirectory() as temp_dir:
            source_cropped_dir = os.path.join(temp_dir, "source_cropped")
            source_mask_dir = os.path.join(temp_dir, "source_mask")
            target_cropped_dir = os.path.join(temp_dir, "target_cropped")
            target_mask_dir = os.path.join(temp_dir, "target_mask")
            
            os.makedirs(source_cropped_dir)
            os.makedirs(source_mask_dir)
            os.makedirs(target_cropped_dir)
            os.makedirs(target_mask_dir)

            # --- Process Source Image ---
            try:
                src_crops, _, _, _ = crop_and_align_face_wrapper([source_image_path])
                if src_crops is None:
                    print(f"No face found in source image {source_image_path}, skipping.")
                    continue
                
                s_crop = src_crops[0].convert("RGB")
                s_pil_im = s_crop.resize((1024,1024), Image.BILINEAR)
                s_mask = faceParsing_demo(faceParsing_model, s_pil_im, convert_to_seg12=opt.seg12, model_name=opt.faceParser_name)
                Image.fromarray(s_mask).save(os.path.join(source_mask_dir, '0.png'))
                s_crop.save(os.path.join(source_cropped_dir, '0.png'))

            except Exception as e:
                print(f"Error processing source image {source_image_path}: {e}")
                continue
                
            # --- Process Target Image ---
            try:
                tgt_crops, orig_images, quads, inv_transforms = crop_and_align_face_wrapper([target_image_path])
                if tgt_crops is None:
                    print(f"No face found in target image {target_image_path}, skipping.")
                    continue
                
                t_crop = tgt_crops[0].convert("RGB")
                orig_image = orig_images[0]
                
                t_pil_im = t_crop.resize((1024,1024), Image.BILINEAR)
                t_mask = faceParsing_demo(faceParsing_model, t_pil_im, convert_to_seg12=opt.seg12, model_name=opt.faceParser_name)
                Image.fromarray(t_mask).save(os.path.join(target_mask_dir, '0.png'))
                t_crop.save(os.path.join(target_cropped_dir, '0.png'))

            except Exception as e:
                print(f"Error processing target image {target_image_path}: {e}")
                continue

            # --- Inference Logic (adapted from one_inference.py) ---
            conf_file = OmegaConf.load(opt.config)
            
            # Create reference tensor from source
            ref_img_path = os.path.join(source_cropped_dir, '0.png')
            img_p_np = cv2.imread(ref_img_path)
            ref_img = cv2.cvtColor(img_p_np, cv2.COLOR_BGR2RGB)
            
            ref_mask_path = os.path.join(source_mask_dir, '0.png')
            ref_mask_img = Image.open(ref_mask_path).convert('L')
            ref_mask_img = np.array(ref_mask_img)

            preserve = conf_file.data.params.test.params['preserve_mask_src_FFHQ']
            ref_mask = np.isin(ref_mask_img, preserve)
            ref_converted_mask = np.zeros_like(ref_mask_img)
            ref_converted_mask[ref_mask] = 255
            ref_converted_mask = Image.fromarray(ref_converted_mask).convert('L')
            
            reference_mask_tensor = get_tensor(normalize=False, toTensor=True)(ref_converted_mask)
            mask_ref = transforms.Resize((224,224))(reference_mask_tensor)
            
            trans = A.Compose([A.Resize(height=224, width=224)])
            ref_img = trans(image=ref_img)['image']
            ref_img = Image.fromarray(ref_img)
            ref_img = get_tensor_clip()(ref_img)
            ref_img = ref_img * mask_ref
            ref_image_tensor = ref_img.to(device, non_blocking=True).to(torch.float16).unsqueeze(0)

            # Create dataset and dataloader for the single target
            test_args = conf_file.data.params.test.params
            test_dataset = VideoDataset(data_path=target_cropped_dir, mask_path=target_mask_dir, **test_args)
            test_dataloader = torch.utils.data.DataLoader(test_dataset, 
                                                batch_size=batch_size, 
                                                num_workers=0, # Set to 0 for simplicity in this script
                                                pin_memory=True, 
                                                shuffle=False)

            start_code = None
            if opt.fixed_code:
                start_code = torch.randn([opt.n_samples, opt.C, opt.H // opt.f, opt.W // opt.f], device=device)

            precision_scope = autocast if opt.precision == "autocast" else nullcontext
            with torch.no_grad():
                with precision_scope("cuda"):
                    with model.ema_scope():
                        for test_batch, prior, test_model_kwargs, segment_id_batch in test_dataloader:
                            uc = None
                            if opt.scale != 1.0:
                                uc = model.learnable_vector.repeat(test_batch.shape[0], 1, 1)
                                if model.stack_feat:
                                    uc2 = model.other_learnable_vector.repeat(test_batch.shape[0], 1, 1)
                                    uc = torch.cat([uc, uc2], dim=-1)

                            landmarks = model.get_landmarks(test_batch) if model.Landmark_cond else None
                            ref_imgs = ref_image_tensor.repeat(test_batch.shape[0], 1, 1, 1)

                            c = model.conditioning_with_feat(ref_imgs.squeeze(1).to(torch.float32), landmarks=landmarks, tar=test_batch.to(device).to(torch.float32)).float()
                            if (model.land_mark_id_seperate_layers or model.sep_head_att) and opt.scale != 1.0:
                                if landmarks is not None:
                                    landmarks = landmarks.unsqueeze(1) if len(landmarks.shape) != 3 else landmarks
                                    uc = torch.cat([uc, landmarks], dim=-1)
                            
                            if c.shape[-1] == 1024: c = model.proj_out(c)
                            if len(c.shape) == 2: c = c.unsqueeze(1)

                            test_model_kwargs = {n: test_model_kwargs[n].to(device, non_blocking=True) for n in test_model_kwargs}
                            z_inpaint = model.encode_first_stage(test_model_kwargs['inpaint_image'])
                            z_inpaint = model.get_first_stage_encoding(z_inpaint).detach()
                            test_model_kwargs['inpaint_image'] = z_inpaint
                            test_model_kwargs['inpaint_mask'] = Resize([z_inpaint.shape[-1], z_inpaint.shape[-1]])(test_model_kwargs['inpaint_mask'])

                            shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                            samples_ddim, _ = sampler.sample(S=steps_for_this_run,
                                                                conditioning=c,
                                                                batch_size=test_batch.shape[0],
                                                                shape=shape,
                                                                verbose=False,
                                                                unconditional_guidance_scale=opt.scale,
                                                                unconditional_conditioning=uc,
                                                                eta=opt.ddim_eta,
                                                                x_T=start_code,
                                                                test_model_kwargs=test_model_kwargs,
                                                                src_im=ref_imgs.squeeze(1).to(torch.float32),
                                                                tar=test_batch.to(device))

                            x_samples_ddim = model.decode_first_stage(samples_ddim)
                            x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                            x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()
                            
                            x_checked_image, _ = check_safety(x_samples_ddim)
                            x_checked_image_torch = torch.from_numpy(x_checked_image).permute(0, 3, 1, 2)
                            
                            # --- Pasting and Saving ---
                            for i, x_sample in enumerate(x_checked_image_torch):
                                x_sample_img = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                                img = Image.fromarray(x_sample_img.astype(np.uint8)).resize((1024,1024), Image.BILINEAR)
                                
                                inv_trans_coeffs = inv_transforms[i]
                                swapped_and_pasted = img.convert('RGBA')
                                pasted_image = orig_image.convert('RGBA')
                                
                                projected = swapped_and_pasted.transform(orig_image.size, Image.PERSPECTIVE, inv_trans_coeffs, Image.BILINEAR)
                                pasted_image.alpha_composite(projected)
                                pasted_image = pasted_image.convert('RGB')
                                
                                src_basename = os.path.splitext(os.path.basename(source_image_path))[0]
                                tgt_basename = os.path.splitext(os.path.basename(target_image_path))[0]
                                
                                result_filename = f"{index:04d}_{tgt_basename}_swapped_with_{src_basename}.png"
                                result_path = os.path.join(image_results_dir, result_filename)
                                pasted_image.save(result_path)

                                end_time_row = time.time()
                                duration_seconds = end_time_row - start_time_row

                                results_data.append({
                                    'source': source_image_path,
                                    'target': target_image_path,
                                    'result': result_path,
                                    'processing_time_seconds': duration_seconds,
                                    'steps_used': steps_for_this_run
                                })
                                break # only one image per dataloader iteration
            # End of with torch.no_grad()
        # End of with tempfile.TemporaryDirectory()
    # End of for loop

    # Save results to CSV
    results_df = pd.DataFrame(results_data)
    output_csv_path = os.path.join(opt.outdir, f"{time_str}_result.csv")
    results_df.to_csv(output_csv_path, index=False)
    
    print(f"\nBatch inference complete.")
    print(f"Results saved to: {opt.outdir}")
    print(f"Manifest CSV saved to: {output_csv_path}")


if __name__ == "__main__":
    main() 
