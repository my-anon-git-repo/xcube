# Steps to run:

Download and data creation:
scripts/launches/download_and_create_zeroshot_dataset.sh

Using hkunlp/instructor-large

1. To train the topic model and topicfy the data 

(done)
launches/topic_modelling_zeroshot.sh --embedding_model_name=hkunlp/instructor-large --base_dir=../tmp2

2. (done) launches/topic_modelling_zeroshot.sh --embedding_model_name=hkunlp/instructor-large --base_dir=../tmp2 --pseudolabel

3. (done)
launches/topic_modelling_zeroshot.sh --embedding_model_name=hkunlp/instructor-large --base_dir=../tmp2 --predict
 ./launches/topic_modelling_zeroshot.sh --embedding_model_name=hkunlp/instructor-large --base_dir=../tmp2 --predict --predict_topk 48 --cross_rerank (/w cross-rerank P@1: 0.2422, w/o P@1: 0.2287 on a split of test 35k, on the full test set /w cross-rerank P@1=24.06 and /w is 22.70)

Performing openai API requests:
launches/topic_modelling_zeroshot.sh --embedding_model_name=hkunlp/instructor-large --base_dir=../tmp2 --predict --prompt_rerank

generates 14 batch files
topk_lbs_cutoff=100, max_records_per_file=10000

firing batches:
./launch_openai_batches.sh api_requests_part1.jsonl [done]
./launch_openai_batches.sh api_requests_part2.jsonl [done]
./launch_openai_batches.sh api_requests_part3.jsonl [done] 
./launch_openai_batches.sh api_requests_part4.jsonl [done]
./launches/launch_openai_batches.sh ../tmp2/LF_AMAZON_131K/api_requests_part5.jsonl [done]
./launches/launch_openai_batches.sh ../tmp2/LF_AMAZON_131K/api_requests_part6.jsonl [done]
./launches/launch_openai_batches.sh ../tmp2/LF_AMAZON_131K/api_requests_part7.jsonl [done]
./launches/launch_openai_batches.sh ../tmp2/LF_AMAZON_131K/api_requests_part8.jsonl [done]
./launches/launch_openai_batches.sh ../tmp2/LF_AMAZON_131K/api_requests_part9.jsonl [done]
./launches/launch_openai_batches.sh ../tmp2/LF_AMAZON_131K/api_requests_part10.jsonl [done]
./launches/launch_openai_batches.sh ../tmp2/LF_AMAZON_131K/api_requests_part11.jsonl [done]
./launches/launch_openai_batches.sh ../tmp2/LF_AMAZON_131K/api_requests_part12.jsonl [done]
./launches/launch_openai_batches.sh ../tmp2/LF_AMAZON_131K/api_requests_part13.jsonl [done]
./launches/launch_openai_batches.sh ../tmp2/LF_AMAZON_131K/api_requests_part14.jsonl [done]

performing the reranking and computing the metrics based on the chatGPT results :
./launches/rerank_gpt_preds.sh
(This works really well, P@1 with chatgpt reranking is 25.82, and without is 24.06)

Using all-MiniLM-L6-v2
1. launches/topic_modelling_zeroshot.sh

2. launches/topic_modelling_zeroshot.sh --pseudolabel

3. launches/topic_modelling_zeroshot.sh --predict
    - Performing Cross Encoder Based reranking 
    - launches/topic_modelling_zeroshot.sh --predict --cross_rerank [done]
    - launches/topic_modelling_zeroshot.sh --predict --predict_topk 48 --cross_rerank (helps P@1 is 19.61 as oppose to without re-ranking 17.65 on a split of 25k test docs)

    - Performing Prompt based Reranking using GPT-4o-mini before making predictions
    - launches/topic_modelling_zeroshot.sh --predict --prompt_rerank [done]

Now the final technique is to train a SOTA dense retrieval model:

GPL Training with margin-mse loss

- `launches/topic_modelling_zeroshot.sh --prep_margin_mseloss_data --num_pos 10`
    Prepare the beir format data
    - (Training) This generates corpus.jsonl, qgen-queries.jsonl and qgen-qrels/train.tsv [ here corpus is the labels, queries are the train data (title + content of the products) and train.tsv is for each query the num_pos many postive labels ] [make sure qgen-queries donot have any null in the text field]
    - (Testing) This also generates queries.jsonl and qrels/test.tsv for evaluation


- `launches/topic_modelling_zeroshot.sh --train_margin_mseloss --negatives_per_query 50 --gpl_steps 140000 --do_evaluation`
    1. Generate hard-negatives.jsonl (for each query there are num_pos many postives and negatives_per_query many negatives)
    2. Generate gpl-training-data.tsv 
    3. Perform gpl training 
    4. omit `--do_evaluation` if you just want to train [Improved $P@1$ from $17.66$ (just topic modelling retrieval) to $19.49$ (margin-mseloss retrieval after gpl style training)]
    5. if `base_dir` has 'tst_predicted.csv' which was produced by topic modelling retrieval then we can aggregate the predictions. This further improves the $P@1$ from 
    $19.49$ to $20.96$  
    6.  - Find evaluation logs of just topic modelling based retrieval at /home/deb/xcube/tmp/LF_AMAZON_131K/logs/default_logfile_2024-11-13_10-46-03.log.
        - Find margin-mseloss training logs at /home/deb/xcube/tmp/LF_AMAZON_131K/logs/default_logfile_2024-11-08_19-13-23.log
        - Find (topic modelling + margin mseloss based retrieval) evaluation log at /home/deb/xcube/tmp/LF_AMAZON_131K/logs/default_logfile_2024-11-12_17-57-19.log



