#!/bin/bash
datasets_folder=$1
prediction_folder=$2

declare -a folders=(
				"falcon-40b-2"
				"falcon-40b-instruct-2"
				"Llama-2-70b-hf"
				"Llama-2-70b-hf-chat"
				"Llama-2-13b-hf"
				"Llama-2-13b-hf-chat"
				"Llama-2-7b-hf"
				"Llama-2-7b-hf-chat"
				"mpt-7b"
				"mpt-7b-instruct"
				)

for key in "${folders[@]}":
do
	echo "Running prediction for ${key}"
	python3 evaluate.py \
		--prediction_folder=${prediction_folder}/${key} \
		--dataset_folder=${datasets_folder} \
		--do_json_conversion
done
