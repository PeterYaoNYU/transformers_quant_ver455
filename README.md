Simply run 
```python
python3 gsm8k_bench.py
```

modify the output dir within the script. 

It runs with a batch size = 1. For some unknown reasons, this version allows only batch = 1 on hqq. 

Quanto is problematic on my machine. 

This version of transformers gives suboptimal performance. 

should check cases where batch size > 1
