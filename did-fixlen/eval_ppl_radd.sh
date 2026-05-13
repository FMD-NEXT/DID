MC=1024
bs=512 

mp=radd-lambda-dce-medium 
wd=./logs/radd-lambda-dce-medium 

loss=lambda_DCE


python evaluation_modeling_radd.py \
--batch_size $bs \
--model_path $mp \
--length 1024 \
--valid_dataset ag_news \
--monte_carlo_timesteps $MC \
--ngpus 8 \
--loss_type $loss \
--work_dir $wd

python evaluation_modeling_radd.py \
--batch_size $bs \
--model_path $mp \
--length 1024 \
--valid_dataset scientific_papers_pubmed \
--monte_carlo_timesteps $MC \
--ngpus 8 \
--loss_type $loss \
--work_dir $wd

python evaluation_modeling_radd.py \
--batch_size $bs \
--model_path $mp \
--length 1024 \
--valid_dataset scientific_papers_arxiv \
--monte_carlo_timesteps $MC \
--ngpus 8 \
--loss_type $loss \
--work_dir $wd

python evaluation_modeling_radd.py \
--batch_size $bs \
--model_path $mp \
--length 1024 \
--valid_dataset wikitext103 \
--monte_carlo_timesteps $MC \
--ngpus 8 \
--loss_type $loss \
--work_dir $wd

python evaluation_modeling_radd.py \
--batch_size $bs \
--model_path $mp \
--length 1024 \
--valid_dataset lambada \
--monte_carlo_timesteps $MC \
--ngpus 8 \
--loss_type $loss \
--work_dir $wd

python evaluation_modeling_radd.py \
--batch_size $bs \
--model_path $mp \
--length 1024 \
--valid_dataset ptb \
--monte_carlo_timesteps $MC \
--ngpus 8 \
--loss_type $loss \
--work_dir $wd

python evaluation_modeling_radd.py \
--batch_size $bs \
--model_path $mp \
--length 1024 \
--valid_dataset lm1b \
--monte_carlo_timesteps $MC \
--ngpus 8 \
--loss_type $loss \
--work_dir $wd