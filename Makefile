PYTHON ?= python3

.PHONY: prepare clean-data index chat evaluate train-dry download-model demo

prepare:
	$(PYTHON) -m src.data.prepare_dataset

clean-data: prepare
	$(PYTHON) -m src.data.clean_dataset

index:
	$(PYTHON) -m src.rag.build_index

chat:
	$(PYTHON) -m src.inference.chat

evaluate:
	$(PYTHON) -m src.evaluation.evaluate

train-dry: clean-data
	$(PYTHON) -m src.training.train_lora

download-model:
	$(PYTHON) -m scripts.download_model

demo: clean-data index evaluate
