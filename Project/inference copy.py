import cv2
import torch
import numpy as np
import clip
from PIL import Image
import json
from torchvision import transforms #added

from Models import SLTModel, convert_to_deploy
from Utils.pose_extractor import PoseKeypointExtractor
from Utils.keypoint_utils import sequence_to_gcn_tensor

# Constants
BOS_TOKEN = 1  # Beginning Of Sequence
EOS_TOKEN = 2  # End Of Sequence
PAD_TOKEN = 0  # Padding
MAX_SENTENCE_LEN = 20  # Max number of glosses to predict per window
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WINDOW_SIZE = 16

def load_model():

    clip_model, _ = clip.load("ViT-B/32", device=DEVICE)

    model = SLTModel(
        clip_model=clip_model,
        adapter_dim=256,
        adapter_layers=range(2, 12),
        fused_dim=512,
        vocab_size=2000
    )

    # 1. Collapse the architecture BEFORE loading the state_dict
    # This turns multi-branch RepBlocks into single Conv2d layers
    convert_to_deploy(model) 
    
    # 2. Now the shapes match the 'slt_model_deployed.pth' file
    model.load_state_dict(torch.load("checkpoints/slt_model_deployed.pth", map_location=DEVICE))
    model.eval()
    return model

def load_vocab(path="checkpoints/vocab.json"):
    with open(path, "r") as f:
        vocab = json.load(f)
    # Create the inverse mapping: index -> word
    return {int(v): k for k, v in vocab.items()}

def preprocess_frames(frames, preprocess):
    processed = []
    for f in frames:
        img = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        img = preprocess(Image.fromarray(img))
        processed.append(img)

    tensor = torch.stack(processed)
    return tensor.unsqueeze(0)

def decode_tokens(tgt_tensor, vocab_mapping=None):
    """
    tgt_tensor: (1, T) tensor of indices
    vocab_mapping: Dictionary {index: "GLOSS_WORD"}
    """
    # Convert tensor to a list of integers
    indices = tgt_tensor.squeeze(0).cpu().numpy().tolist()
    
    # Filter out special tokens (BOS, EOS, PAD)
    meaningful_indices = [
        idx for idx in indices 
        if idx not in [BOS_TOKEN, EOS_TOKEN, PAD_TOKEN]
    ]
    
    # Map indices to words
    if vocab_mapping:
        words = [vocab_mapping.get(idx, "<UNK>") for idx in meaningful_indices]
        return " ".join(words)
    else:
        # Fallback if vocab is not loaded
        return f"Indices: {meaningful_indices}"

def run_inference():
    model = load_model()
    vocab = load_vocab()

    pose_extractor = PoseKeypointExtractor()
    cap = cv2.VideoCapture(0)
    
    frame_buffer = []
    kp_buffer = []

    # added: Mimic the training spatial transformations
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
    ])
    #-----------
    while True:
        ret, frame = cap.read()
        if not ret: break

        # 1. Extract and Buffer
        kp = pose_extractor.extract_keypoints(frame)
        frame_buffer.append(frame)
        kp_buffer.append(kp)

        # 2. Process when buffer is full (e.g., 16 or 32 frames)
        if len(frame_buffer) >= WINDOW_SIZE:
            #rgb_tensor = preprocess_frames(frame_buffer, model.rgb_encoder.clip_preprocess).to(DEVICE) 
            rgb_tensor = preprocess_frames(frame_buffer, preprocess).to(DEVICE) #added: In inference.py, you reference model.rgb_encoder.clip_preprocess. However, in clip.py there is no such attribute exposed. You need to recreate the Torchvision transform pipeline locally inside the inference file to avoid an AttributeError.
            # Shape (1, 2, T, 105)
            key_tensor = sequence_to_gcn_tensor(kp_buffer).to(DEVICE)

            with torch.no_grad():
                out = model(rgb_tensor, key_tensor)
                
                # Use Boundary Head to see if movement is a valid sign
                is_sign = out['boundary_probs'].item() > 0.5
                
                if is_sign:
                    # Use Translator (greedy decoding)
                    # Start with BOS token
                    tgt = torch.tensor([[BOS_TOKEN]]).to(DEVICE)
                    for _ in range(MAX_SENTENCE_LEN):
                        logits = model.translator(out['context_seq'], tgt)
                        next_token = torch.argmax(logits[:, -1, :], dim=-1)
                        tgt = torch.cat([tgt, next_token.unsqueeze(0)], dim=1)
                        if next_token == EOS_TOKEN: break
                    
                    translation = decode_tokens(tgt, vocab_mapping=vocab)
                    print("Translated Sign:", translation)

            # Reset buffers
            frame_buffer, kp_buffer = [], []

if __name__ == "__main__":

    run_inference()