### Generate the data

asub_cpu PYTHONPATH=. python src --data-paths 

./data/math_hard/qna_rits.yaml



### Move the data to initial location

`ver=112524`



`mkdir -p output/math_hard_rits/$ver/00_init_gen/`



`mv output/math_hard_rits/data.jsonl 

output/math_hard_rits/$ver/00_init_gen/data.jsonl`



### Format the generated data to conversation format

`ver=112524`

`mkdir -p output/math_hard_rits/$ver/01_formatted/`



`PYTHONPATH=. python src/databuilders/generation/math_hard/math_hard_output_converter.py 

output/math_hard_rits/$ver/00_init_gen/data.jsonl 

output/math_hard_rits/$ver/01_formatted/data.jsonl`



### Tagging the data

`mkdir -p output/math_hard_rits/$ver/02_magpie/`



Cnage the version in the config (need to find a way to do it with command line)



`vi ./data/transformation/magpie/tagging/qna_math_hard.yaml`



`asub_40 PYTHONPATH=. python -m src.__main__ --data-path 

./data/transformation/magpie/tagging/qna_math_hard.yaml 

--seed-batch-size 50`



`ljq output/math_hard_rits/112624/02_magpie/magpie_tag_transform/data.jsonl | grep \"excellent\" | wc`



### calculate distances

`vi ./data/transformation/magpie/distance/qna_math_hard.yaml`



`asub_40 PYTHONPATH=. python -m src.__main__ --data-path 

./data/transformation/magpie/distance/qna_math_hard.yaml 

--seed-batch-size 50`





### Filtering the data



`vi data/transformation/magpie/filtering/qna_math_hard.yaml`



`asub_40 PYTHONPATH=. python -m src.__main__ --data-path 

./data/transformation/magpie/filtering/qna_math_hard.yaml 

--seed-batch-size 50`