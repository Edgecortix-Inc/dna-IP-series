#!/usr/bin/env python3
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import cv2
import time
import kivy
import ip_runtime
import torch
import numpy as np

from kivy.app import App
from kivy.uix.image import Image
from kivy.uix.boxlayout import BoxLayout
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.uix.label import Label 


class KivyCamera(Image):
    def __init__(self, capture, fps, **kwargs):
        super(KivyCamera, self).__init__(**kwargs)
        self.capture = capture
        Clock.schedule_interval(self.update, 1.0 / fps)

    def update(self, dt):
        ret, frame = self.capture.read()
        if ret:
            
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # forward
            t0 = time.time()
            boxes, labels, probs = predict(img)
            fps = round(1 / (time.time() - t0),2)
            for i in range(boxes.size(0)):
                box = boxes[i, :]
                label = "PERSON"
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 4)

                cv2.putText(frame, label,
                        (box[0]-20, box[1]-20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,  # font scale
                        (255, 255, 255),
                        2, cv2.LINE_AA)  # line type
                cv2.putText(frame, "FPS: "+str(fps), (20,30),
                         cv2.FONT_HERSHEY_SIMPLEX,
                         1, (255,255,255), 2, cv2.LINE_AA)
 
            # convert it to texture
            buf1 = cv2.flip(frame, 0)
            buf = buf1.tostring()
            image_texture = Texture.create(
                size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            image_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            # display image from the texture
            self.texture = image_texture
            

class CamApp(App):
    def build(self):
        self.capture = cv2.VideoCapture(0)
        self.capture.set(3,1280)
        self.capture.set(4,720)
        self.my_camera = KivyCamera(capture=self.capture, fps=30)
        labelTitle= Label(text="[b]Dynamic Neural Accelerator™[/b] Person Detection on [b]Xilinx Alveo™ U50[/b]\n, Model = Mobilenetv2-SSD, Precision = INT8", 
                          size_hint=(0.2,0.3), halign="center",pos_hint={'center_x':0.5, 'center_y':2}, markup=True, font_size=28)
        #im1 = Image(source="", keep_ratio=True, size_hint_x=0.3, size_hint_y=0.3, opacity=1)
        layout = BoxLayout(orientation='vertical')
        layout.add_widget(labelTitle)
        layout.add_widget(self.my_camera)
        #layout.add_widget(im1)
       
        return layout

    def on_stop(self):
        #without this, app will not exit even if the window is closed
        self.capture.release()	


def predict(image):
    height, width, _ = image.shape

    image = cv2.resize(image, (640, 480))
    mean = np.array([127, 127, 127], dtype=np.float32)
    image = image.astype(np.float32)
    image -= mean
    image = image.astype(np.float32)
    image = image / 128.0
    image = np.expand_dims(image, axis=0)

    outputs = iprt.Run(image)
    scores = torch.from_numpy(outputs[0].reshape(3000, 6))
    boxes = torch.from_numpy(outputs[1].reshape(3000, 4))

    prob_threshold = 0.5
    iou_threshold = 0.45
    sigma = 0.5
    top_k = 10
    candidate_size = 200

    picked_box_probs = []
    picked_labels = []
    class_ids = list(range(1, scores.size(1)))
    for class_index in class_ids:
        probs = scores[:, class_index]
        mask = probs > prob_threshold
        probs = probs[mask]
        if probs.size(0) == 0:
            continue
        subset_boxes = boxes[mask, :]
        box_probs = torch.cat([subset_boxes, probs.reshape(-1, 1)], dim=1)

        box_probs = hard_nms(box_probs,
                             score_threshold=prob_threshold,
                             iou_threshold=iou_threshold,
                             sigma=sigma,
                             top_k=top_k,
                             candidate_size=candidate_size)

        picked_box_probs.append(box_probs)
        picked_labels.extend([class_index] * box_probs.size(0))
    if not picked_box_probs:
        return torch.tensor([]), torch.tensor([]), torch.tensor([])
    picked_box_probs = torch.cat(picked_box_probs)
    picked_box_probs[:, 0] *= width
    picked_box_probs[:, 1] *= height
    picked_box_probs[:, 2] *= width
    picked_box_probs[:, 3] *= height
    return picked_box_probs[:, :4], torch.tensor(picked_labels), picked_box_probs[:, 4]


def hard_nms(box_scores, score_threshold=None, iou_threshold=None, sigma=0.5, top_k=-1, candidate_size=200):
    def area_of(left_top, right_bottom) -> torch.Tensor:
        hw = torch.clamp(right_bottom - left_top, min=0.0)
        return hw[..., 0] * hw[..., 1]

    def iou_of(boxes0, boxes1, eps=1e-5):
        overlap_left_top = torch.max(boxes0[..., :2], boxes1[..., :2])
        overlap_right_bottom = torch.min(boxes0[..., 2:], boxes1[..., 2:])
    
        overlap_area = area_of(overlap_left_top, overlap_right_bottom)
        area0 = area_of(boxes0[..., :2], boxes0[..., 2:])
        area1 = area_of(boxes1[..., :2], boxes1[..., 2:])
        return overlap_area / (area0 + area1 - overlap_area + eps)

    scores = box_scores[:, -1]
    boxes = box_scores[:, :-1]
    picked = []
    _, indexes = scores.sort(descending=True)
    indexes = indexes[:candidate_size]
    while len(indexes) > 0:
        current = indexes[0]
        picked.append(current.item())
        if 0 < top_k == len(picked) or len(indexes) == 1:
            break
        current_box = boxes[current, :]
        indexes = indexes[1:]
        rest_boxes = boxes[indexes, :]
        iou = iou_of(
            rest_boxes,
            current_box.unsqueeze(0),
        )
        indexes = indexes[iou <= iou_threshold]

    return box_scores[picked, :]


iprt = ip_runtime.IPRuntime()
iprt.Setup("mv2ssd")

def main():    
    from kivy.core.window import Window
    Window.fullscreen=False
    Window.maximize()
    CamApp().run()


if __name__ == '__main__':
    main()
