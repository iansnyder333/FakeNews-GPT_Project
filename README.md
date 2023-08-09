# GPT, T5-Peft Project

## Intuition
Fine-tuning large scale pre-trained transformers leads to huge performance gains, and offers a paradigm to achieve state of the art performance in NLP tasks at a fraction of the cost compared to building an LLM from scratch. However, as modern pre-trained transformers scale in size to trillions of parameters, the ability to store and fine-tune without a high-performance computer becomes impossible. This issue has been addressed with PEFT, Parameter-Efficient Fine-Tuning. Which only fine-tunes a small number of (extra) model parameters while freezing most parameters of the pretrained LLMs, thereby greatly decreasing the computational and storage costs. This project demonstrates the use of Low-Rank Adaptation of LLMs using HuggingFace and PyTorch to fine tune Flan-T5-Large (~900M Parameters) to summarize content using consumer hardware, easily running on google-collab. The project also uses a scratch built GPT, along with a fine-tuned GPT-2 for content generation that can be used independently or for summarization tests. 

**The purpose of this project was for me to get hands on exposure using LLMS for NLP tasks in a practical setting**

##
<https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/peft/lora_diagram.png>
