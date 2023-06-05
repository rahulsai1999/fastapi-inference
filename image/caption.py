import torch

from transformers import AutoProcessor, AutoModelForCausalLM

device = "cuda" if torch.cuda.is_available() else "cpu"
processor = AutoProcessor.from_pretrained("microsoft/git-large-textcaps")
model = AutoModelForCausalLM.from_pretrained("microsoft/git-large-textcaps")

model.to(device)


def generate_caption(processor, model, image, tokenizer=None, use_float_16=False):
    inputs = processor(images=image, return_tensors="pt").to(device)

    if use_float_16:
        inputs = inputs.to(torch.float16)

    generated_ids = model.generate(pixel_values=inputs.pixel_values, max_length=50)

    if tokenizer is not None:
        generated_caption = tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]
    else:
        generated_caption = processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]

    return generated_caption


def generate_captions(image):
    caption_git_large_textcaps = generate_caption(processor, model, image)
    return caption_git_large_textcaps
