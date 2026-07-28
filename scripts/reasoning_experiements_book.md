# Reasoning Model Experiments Logbook

## Setup

```sh
# 📥 Download and run the setup script
curl -L -o setup_xcube.sh https://tinyurl.com/setup-xcube
chmod u+x setup_xcube.sh
bash setup_xcube.sh
```

### Optional: SSH Key Setup
```sh
# 🔐 Upload SSH key to GitHub (optional)
curl -L -o upload_ssh_key_to_github.sh https://tinyurl.com/upload-ssh
bash upload_ssh_key_to_github.sh
```

##  Prepare the dataset

```sh
(base) deb@deb-GIGABYTE:~/xcube/scripts$
launches/download_and_create_zeroshot_dataset_new.sh \
    --source_url=XURLs.LF_WIKISEEALSO_320K
```

This will create the files:
```sh
(base) deb@deb-GIGABYTE:~/.xcube/data$ tree LF-WikiSeeAlso-320K
LF-WikiSeeAlso-320K
├── all_pairs.txt
├── lbl.json
├── lf-wikiseealso-320k.csv
├── trn.json
└── tst.json

0 directories, 5 files
```

Optional: If we want to store the dataset at aws

```sh
(base) deb@deb-GIGABYTE:~/.xcube/data$
tar -czf LF-WikiSeeAlso-320K.tar.gz LF-WikiSeeAlso-320K/
aws s3 cp LF-WikiSeeAlso-320K.tar.gz s3://xcubebucket/lf-wikiseealso-320k/
```

Then to create a sample run:
```sh
(base) deb@deb-GIGABYTE:~/xcube/scripts$ ./launches/launch_reason_sample_dset_create.sh 

(base) deb@deb-GIGABYTE:~/.xcube/data$ tree LF-WikiSeeAlso-320K_sample
LF-WikiSeeAlso-320K_sample
├── lf-wikiseealso-320k_sample.csv
├── lf-wikiseealso-320k_sample_unique_labels.csv
└── report.txt # contains the sampling stats

0 directories, 3 files
```

Optional: If we want to store the sample dataset at aws:
```sh
(base) deb@deb-GIGABYTE:~/.xcube/data$ 
tar -czf LF-WikiSeeAlso-320K_sample.tar.gz LF-WikiSeeAlso-320K_sample/
aws s3 mv LF-WikiSeeAlso-320K_sample.tar.gz s3://xcubebucket/lf-wikiseealso-320k/
```



Next, to create reasoning prompts for the dataset created above:
```sh
(base) deb@deb-GIGABYTE:~/xcube/scripts$ ./launches/launch_reason_prompts_generate.sh
```
This will create
1. `lf-wikiseealso-320k_sample_prompts.jsonl`  # file containing the prompts
2. `prompt_test.py` # a test prompt for quick manual checking with a reasoning model

```sh
(base) deb@deb-GIGABYTE:~/.xcube/data$ tree LF-WikiSeeAlso-320K_sample/
LF-WikiSeeAlso-320K_sample/
├── lf-wikiseealso-320k_sample.csv
├── lf-wikiseealso-320k_sample_prompts.jsonl #1
├── lf-wikiseealso-320k_sample_unique_labels.csv
├── prompt_test.py #2
└── report.txt

0 directories, 5 files
```