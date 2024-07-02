import json, os, tqdm
import jittor as jt
import datetime
from JDiffusion.pipelines import StableDiffusionPipeline

max_num = 15
dataset_root = "/root/autodl-tmp/A/"

model_id = "stabilityai/stable-diffusion-2-1"

with jt.no_grad():
    for tempid in tqdm.tqdm(range(0, max_num)):
        taskid = "{:0>2d}".format(tempid)
        pipe = StableDiffusionPipeline.from_pretrained(model_id).to("cuda")
        # pipe = StableDiffusionPipeline.from_pretrained(model_id, cache_dir="").to("cuda")
        pipe.load_lora_weights(f"/root/autodl-tmp/result_1200_rank8/style_{taskid}")

        # load json
        with open(f"{dataset_root}/{taskid}/prompt.json", "r") as file:
            prompts = json.load(file)

        for id, prompt in prompts.items():
            print(prompt)
            image = pipe(prompt + f" in style_{taskid}", num_inference_steps=25, width=512, height=512).images[0]
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%s")
            os.makedirs(f"/root/autodl-tmp/result/test/output/{taskid}", exist_ok=True)
            image.save(f"/root/autodl-tmp/result/test/output/{taskid}/{prompt}.png")
