口罩模型辨識

利用kaggle上的資源，用keras訓練出一個model

再使用這個model來作為是否佩戴口罩的依據

我們使用電腦攝像頭作為影像來源

當畫面中沒有人物的時候，會呈現粉色框框

當畫面中有人但是沒有佩戴口罩時，會呈現紅色框框

當畫面中有人且有正確佩戴口罩，會呈現綠色框框

當畫面中有人，但卻沒有正確佩戴口罩，會呈現藍色框框

Training model 從這裡下載：
https://drive.google.com/file/d/1LGLsZXhoWAtv7sJGlLWXPsDpyhIuSf34/view?usp=sharing

requirements: requirements.txt
