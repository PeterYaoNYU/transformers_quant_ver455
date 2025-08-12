### Reproducibility

Clone the main branch, install from source, install hqq with pip.   

Run the gsm8k data parallel bnenchmark on the first 128 test cases of GSM8K  

```python
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 gsm8k_batch_sharded_dp.py 
```

Check again by eval the merged result

```python
python3 eval_single_gsm8k.py <result_file_path>
```

Turn on Quant, change to
```python
USE_QUANT_CACHE=True
```

Do the reverse to do no KV quant. 


