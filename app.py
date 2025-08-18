
import gradio as gr
import os
import torch

from model import create_effnet_model
from timeit import default_timer as timer
from typing import Tuple, Dict

class_names = ['pizza', 'steak', 'sushi']

effnet, effnet_transforms = create_effnet_model(num_classes=3,
                                                 seed=42)

effnet.load_state_dict(torch.load(f='02_pretrained_effnet_model.pth',
                                  map_location=torch.device('cpu')))

def predict(img) -> Tuple[Dict, float]:
  start_time = timer()

  img = auto_transforms(img).unsqueeze(0)

  effnet.eval()
  with torch.inference_mode():
    pred_probs = torch.softmax(effnet(img), dim=1)

  pred_labels_and_probs = {class_names[i]: float(pred_probs[0][i]) for i in range(len(class_names))}
  pred_time = round(timer()-start_time, 5)
  return pred_labels_and_probs, pred_time

  ## app

title = 'FoodVision app 🍕🥩🍣'
description = 'An EfficientNet Computer Vision Model to Classify Images of Food as Pizza, Steak or Sushi.'
article = 'Created at Pytorch Course'

examples_list = [['examples/' + example] for example in os.listdir('examples')]

demo = gr.Interface(fn=predict,
                    inputs=gr.Image(type='pil'),
                    outputs=[gr.Label(num_top_classes=3, label='Predictions'),
                             gr.Number(label='Prediction time (s)')],
                    examples=examples_list,
                    title=title,
                    description=description,
                    article=article
)

demo.launch()
