import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import tempfile
import pandas as pd
import cv2
import torch
import numpy as np
import tempfile
from time import sleep
import time
import sqlite3
import datetime
import locale

locale.setlocale(locale.LC_ALL, '')

model = YOLO('../best_data_logo_1.pt')

cap = cv2.VideoCapture(0)
start_line_A = (0, 0)
end_line_A = (320, 480)
counter_A = 0



st.title("DEEP LEARNING FINAL PROJECT")
secim=st.selectbox('Class', ['apple', 'bmw', 'carlsberg', 'cocacola', 'corona', 'dhl', 'fedex', 'ferrari', 'heineken_text', 'milka', 'nike', 'nivea', 'pepsi', 'shell', 'starbucks'])
brands = ['apple', 'bmw', 'carlsberg', 'cocacola', 'corona', 'dhl', 'fedex', 'ferrari', 'heineken_text', 'milka', 'nike', 'nivea', 'pepsi', 'shell', 'starbucks']
index = brands.index(secim)


st.markdown(index)


frame_placeholder = st.empty()
stop_button = st.button("Stop")


st.title("Video Yakalama")


while cap.isOpened() and not stop_button:
    ret, frame = cap.read()
    frame=cv2.resize(frame, (640, 480))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    success, frame = cap.read()


    if ret:
        results = model.predict(frame,conf=0.80)

        nsn_tspt = results[0].plot()
        radius = 3
        color = (255,0,0)
        thickness = 2
        empty_tensor = torch.tensor([], device='cuda:0')
        tt=results[0].boxes.cls
        sonuc=torch.equal(tt, empty_tensor)
        if sonuc==False:
            for result in results:
                bboxes = []
                confidences = []
                class_ids = []
                for data in result.boxes.data.tolist():
                    x1, y1, x2, y2, basarim, class_id = data
                    x1 = int(x1)
                    y1 = int(y1)
                    x2 = int(x2)
                    y2 = int(y2)
                    etiket = int(class_id)
                    print(etiket)
                    st.markdown(etiket)
                    
                    textt = str(len(results[0].boxes.data))
                    cv2.putText(nsn_tspt, textt,(10,100),cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (252, 140, 3), 2)

                    
                    if etiket == index:
                        
                        tehlike_text = f'DIKKAT!!!! Tehlikeli bolgede {textt} kisi tespit edildi'
                        cv2.putText(nsn_tspt, tehlike_text, (50, 700), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                                    (252, 140, 3), 2)
                        output = nsn_tspt.copy()

                        roi = nsn_tspt[y1:y1+y2, x1:x1+x2]
                        # Belirli bölgeyi bulanıklaştır
                        blurred_roi = cv2.GaussianBlur(roi, (31, 31), sigmaX=10)

                        # Bulanıklaştırılmış bölgeyi orijinal resme yerleştir
                        nsn_tspt[y1:y1+y2, x1:x1+x2] = blurred_roi

                    else:
                        print("yok")

        nsn_tspt = cv2.cvtColor(nsn_tspt, cv2.COLOR_BGR2RGB)
        nsn_tspt = cv2.flip(nsn_tspt, 1)
        frame_placeholder.image(nsn_tspt,caption = "YOLOv8 Inference",channels="RGB")
        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


cap.release()
cv2.destroyAllWindows()
