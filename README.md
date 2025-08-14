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


```
rm -rf mkdir wrong_quant_math_nocot 
rm -rf correct_quant_math_nocot
mkdir wrong_quant_math_nocot 
cp mg_out_longer_quant_math_150_nocot/*merged.jsonl wrong_quant_math_nocot/

python3 copy_correct_answers_folder.py --wrong-dir wrong_quant_math_nocot/ --correct-dir math_sampled_dataset/ --out-dir correct_quant_math_nocot

python3 evaluate_math_new.py correct_quant_math_nocot/





rm -rf mkdir wrong_noquant_math_nocot 
rm -rf correct_noquant_math_nocot
mkdir wrong_noquant_math_nocot 
cp mg_out_longer_noquant_math_150_nocot/*merged.jsonl wrong_noquant_math_nocot/

python3 copy_correct_answers_folder.py --wrong-dir wrong_noquant_math_nocot/ --correct-dir math_sampled_dataset/ --out-dir correct_noquant_math_nocot

python3 evaluate_math_new.py correct_noquant_math_nocot/





mkdir wrong_quant_math_cot 
cp mg_out_longer_quant_math_150/*merged.jsonl wrong_quant_math_cot/

python3 copy_correct_answers_folder.py --wrong-dir wrong_quant_math_cot/ --correct-dir math_sampled_dataset/ --out-dir correct_quant_math_cot

python3 evaluate_math_new.py correct_quant_math_cot/



mkdir wrong_noquant_math_cot 
cp mg_out_longer_noquant_math_150/*merged.jsonl wrong_noquant_math_cot/

python3 copy_correct_answers_folder.py --wrong-dir wrong_noquant_math_cot/ --correct-dir math_sampled_dataset/ --out-dir correct_noquant_math_cot

python3 evaluate_math_new.py correct_noquant_math_cot/
```