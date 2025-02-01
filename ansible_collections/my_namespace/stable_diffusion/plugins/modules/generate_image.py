from ansible.module_utils.basic import AnsibleModule
import torch
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, EulerDiscreteScheduler
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
import os


def generate_image(model_name, prompt, guidance_scale, steps, seed, width, height, output_location, file_type):
    base = "stabilityai/stable-diffusion-xl-base-1.0"
    repo = "ByteDance/SDXL-Lightning"
    ckpt = "sdxl_lightning_4step_unet.safetensors" # Use the correct ckpt for your step setting!

    unet = UNet2DConditionModel.from_config(base, subfolder="unet").to("cuda", torch.float16)
    unet.load_state_dict(load_file(hf_hub_download(repo, ckpt), device="cuda"))
    pipe = StableDiffusionXLPipeline.from_pretrained(base, unet=unet, torch_dtype=torch.float16, variant="fp16").to("cuda")
    
    if model_name == "sdxl-lightning":
        pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
        
    generator = torch.Generator("cuda").manual_seed(seed) if seed is not None else None
    
    image = pipe(
        prompt,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        width=width,
        height=height,
        generator=generator
    ).images[0]

    full_path = f"{output_location}.{file_type}"
    image.save(full_path)
    
    return full_path

def main():
    module_args = dict(
        model_name=dict(type='str', required=True),
        prompt=dict(type='str', required=True),
        guidance_scale=dict(type='float', default=7.5),
        steps=dict(type='int', default=50),
        seed=dict(type='int', default=None),
        width=dict(type='int', default=1024),
        height=dict(type='int', default=1024),
        output_location=dict(type='str', required=True),
        file_type=dict(type='str', choices=['png', 'jpeg'], default='png')
    )
    
    module = AnsibleModule(
        argument_spec=module_args,
        supports_check_mode=False
    )

    try:
        result_image_path = generate_image(
            model_name=module.params['model_name'],
            prompt=module.params['prompt'],
            guidance_scale=module.params['guidance_scale'],
            steps=module.params['steps'],
            seed=module.params['seed'],
            width=module.params['width'],
            height=module.params['height'],
            output_location=module.params['output_location'],
            file_type=module.params['file_type']
        )
        
        module.exit_json(changed=True, result_image_path=result_image_path)
    except Exception as e:
        module.fail_json(msg=f"An error occurred while generating the image: {e}")

if __name__ == '__main__':
    main()
