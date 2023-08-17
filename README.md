# GPT, T5-Peft Project

![lora_diagram](https://github.com/iansnyder333/FakeNews-GPT_Project/assets/58576523/491c33b5-741b-4390-90a3-3f9ddcc6b08a)
source: <https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/peft/lora_diagram.png>

## Intuition
Fine-tuning large scale pre-trained transformers leads to huge performance gains, and offers a paradigm to achieve state of the art performance in NLP tasks at a fraction of the cost compared to building an LLM from scratch. However, as modern pre-trained transformers scale in size to trillions of parameters, the ability to store and fine-tune without a high-performance computer becomes impossible. This issue has been addressed with PEFT, Parameter-Efficient Fine-Tuning. Which only fine-tunes a small number of (extra) model parameters while freezing most parameters of the pretrained LLMs, thereby greatly decreasing the computational and storage costs. This project demonstrates the use of Low-Rank Adaptation of LLMs using HuggingFace and PyTorch to fine tune Flan-T5-Large (~900M Parameters) to summarize content using consumer hardware, easily running on google-collab. The project also uses a scratch built GPT, along with a fine-tuned GPT-2 for content generation that can be used independently or for summarization tests. 

**The purpose of this project was for me to get hands on exposure using LLMS for NLP tasks in a practical setting**

## Table of Contents

- [How to Run and Install](#how-to-run-and-install)
- [Video Demonstration](#video-demonstration)
- [Notes](#notes)

## How to Run and Install

Install (MacOS)
```sh
git clone https://github.com/iansnyder333/FakeNews-GPT_Project.git
cd FakeNews-GPT_Project
python3.11 -m venv venv
source venv/bin/activate
pip3.11 install -r requirements.txt
```
Run (MacOS)
```sh
#Generate News
python3.11 streamlit run FakeNewsApp.py
#Summary
python3.11 streamlit run SummaryApp.py
```

## Video Demonstration

Generate News

https://github.com/iansnyder333/FakeNews-GPT_Project/assets/58576523/fcbab6a8-97b2-45cf-8dd2-f7ddace74c9f

Summarize Text



https://github.com/iansnyder333/FakeNews-GPT_Project/assets/58576523/a88ea87f-4ea5-419d-ada2-00811bf6778f



## Notes 

Performance for the Flan-T5 Summary Model
<img width="746" alt="FT5-Eval" src="https://github.com/iansnyder333/FakeNews-GPT_Project/assets/58576523/5a6f4d30-9e71-4482-b32e-a46b2a2964ef">

Training scripts are all located in their respective config files. I will have visuals uploaded to readme in a future commit.


