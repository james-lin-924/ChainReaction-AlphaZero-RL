1. 載入必要模組:
```bash
pip install -r requirements.txt
```

2. 建立C++ 模組:
```bash
python setup.py build_ext --inplace
```

3. 進行訓練:
```bash
python train_loop.py
```

4. 測試訓練結果(可忽略):
```bash
python evaluation.py
```

5. Gradio執行:
```bash
python app.py
```


**詳細資訊:**

https://hackmd.io/q31JDV-qQjG8VP4X75THVA?edit