import sys
import cv2
import numpy as np
import argparse
import json

sys.path.append(sys.path[0]+"/tracker")
sys.path.append(sys.path[0]+"/tracker/model")

from track_anything import TrackingAnything
from download_models import download_checkpoint


def get_prompt(click_state):
    return {
        "prompt_type":["click"],
        "input_point":click_state[0],
        "input_label":click_state[1],
        "multimask_output":"True",
    }


def process_video(input_path, output_video_path, output_mask_path, model_type, click_state):
    folder = './checkpoints'

    SAM_checkpoint_dict = {
        'vit_h': "sam_vit_h_4b8939.pth",
        'vit_l': "sam_vit_l_0b3195.pth",
        "vit_b": "sam_vit_b_01ec64.pth"
    }
    SAM_checkpoint_url_dict = {
        'vit_h': "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
        'vit_l': "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
        'vit_b': "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
    }

    SAM_checkpoint = download_checkpoint(SAM_checkpoint_url_dict[model_type], folder, SAM_checkpoint_dict[model_type])
    xmem_checkpoint = download_checkpoint('https://github.com/hkchengrex/XMem/releases/download/v1.0/XMem-s012.pth', folder, 'XMem-s012.pth')

    class Args:
        device = "cuda:0"
        sam_model_type = model_type

    model = TrackingAnything(SAM_checkpoint, xmem_checkpoint, None, Args())

    frames = []

    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    model.samcontroler.sam_controler.reset_image()
    model.samcontroler.sam_controler.set_image(frames[0])
    prompt = get_prompt(click_state=click_state)

    mask, logit, painted_image = model.first_frame_click(
        image=frames[0],
        points=np.array(prompt["input_point"]),
        labels=np.array(prompt["input_label"]),
        multimask=prompt["multimask_output"],
    )

    # painted_image.show()

    model.xmem.clear_memory()

    masks, logits, painted_images = model.generator(images=frames, template_mask=mask)
    model.xmem.clear_memory()

    if output_mask_path:
        np.savez_compressed(output_mask_path, *masks)

    if output_video_path:
        video_writer = cv2.VideoWriter(
            output_video_path,
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps=fps,
            frameSize=(frame_width, frame_height)
        )

        for output_frame in painted_images:
            video_writer.write(cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR))

        video_writer.release()


def main():
    parser = argparse.ArgumentParser(description="Process a video with a specified model type.")
    parser.add_argument('input_video', type=str, help="Path to the input video file")
    parser.add_argument('--output_video', type=str, help="Path to save the output video file (optional)")
    parser.add_argument('--output_mask', type=str, help="Path to save the output mask file (optional)")
    parser.add_argument('model_type', type=str, help="vit_h (higher mem) or vit_b (lower mem)")
    parser.add_argument('click_state', type=str, help="Click state as a nested list")

    args = parser.parse_args()

    click_state = json.loads(args.click_state.strip("'"))

    process_video(args.input_video, args.output_video, args.output_mask, args.model_type, click_state)


if __name__ == "__main__":
    main()
