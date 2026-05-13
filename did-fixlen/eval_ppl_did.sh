MC=1024 
bs=512

mp=$1 # model path, for example, ../output/2025.08.20/021412
wd=./logs/did

loss=DICE

python evaluation_modeling_did.py \
--batch_size $bs \
--model_path $mp \
--length 1024 \
--valid_dataset ag_news \
--monte_carlo_timesteps $MC \
--ngpus 8 \
--loss_type $loss \
--work_dir $wd

python evaluation_modeling_did.py \
--batch_size $bs \
--model_path $mp \
--length 1024 \
--valid_dataset scientific_papers_pubmed \
--monte_carlo_timesteps $MC \
--ngpus 8 \
--loss_type $loss \
--work_dir $wd

python evaluation_modeling_did.py \
--batch_size $bs \
--model_path $mp \
--length 1024 \
--valid_dataset scientific_papers_arxiv \
--monte_carlo_timesteps $MC \
--ngpus 8 \
--loss_type $loss \
--work_dir $wd

python evaluation_modeling_did.py \
--batch_size $bs \
--model_path $mp \
--length 1024 \
--valid_dataset wikitext103 \
--monte_carlo_timesteps $MC \
--ngpus 8 \
--loss_type $loss \
--work_dir $wd

python evaluation_modeling_did.py \
--batch_size $bs \
--model_path $mp \
--length 1024 \
--valid_dataset lambada \
--monte_carlo_timesteps $MC \
--ngpus 8 \
--loss_type $loss \
--work_dir $wd

python evaluation_modeling_did.py \
--batch_size $bs \
--model_path $mp \
--length 1024 \
--valid_dataset ptb \
--monte_carlo_timesteps $MC \
--ngpus 8 \
--loss_type $loss \
--work_dir $wd

python evaluation_modeling_did.py \
--batch_size $bs \
--model_path $mp \
--length 1024 \
--valid_dataset lm1b \
--monte_carlo_timesteps $MC \
--ngpus 8 \
--loss_type $loss \
--work_dir $wd