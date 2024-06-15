# MATCH
Measure Alignment Through Comparative Human Judgement for FCM Extraction
---

## Description
This project seeks to fine-tune LLMs for the task of causal relation extraction. Datum will consist of short passages (e.g., sentences) from scholarly documents such as research articles. Passages are associated to signed, directed edges of a causal relation graph. These edges are encoded in text responses and serve as targets for next-token prediction. There's a lot ot unpack here, but don't be intimidated! We'll discuss each concept in turn.

A key prerequisite for successful fine-tuning is possession of a high-quality, labeled dataset. Hence, **Project Phase I** will include the development and curation of such a dataset. Next, using state-of-the-art fine-tuning methods, including LoRA and 4-bit quantization, the project team will adapt off-the-shelf LLMs using this dataset for **Project Phase II**.

### A Note on this Repo

The code in this repository is research-level and intended to partially seed the project. The final project may diverge from the specific methods herein.

## Getting Started
____

### Dependencies

* We will discuss and list dependencies during our initial meetings.

### Installing

* TBD

### Executing program

* TBD

## Authors
____
Contributors names and contact info:
XXXX

## Version History
____
* 0.1
    * Initial Release

## Resources
____
Here is a collection of resources that you might find helpful, interesting, inspiring.

* Andrej Karpathy's excellent videos on language models and building GPT from scratch:
  * [The spelled-out intro to language modeling: building makemore](https://www.youtube.com/watch?v=PaCmpygFfXo)
  * [Let's build GPT: from scratch, in code, spelled out.](https://youtu.be/kCc8FmEb1nY?si=9BTDenI-fPg_pb9P)

* Various resources discussing LoRA and quantization for fine-tuning LLMs
  * [PEFT LoRA Explained in Detail - Fine-Tune your LLM on your local GPU](https://www.youtube.com/watch?v=YVU5wAA6Txo)
  * [Understanding 4bit Quantization: QLoRA explained (w/ Colab)](https://www.youtube.com/watch?v=TPcXVJ1VSRI)
  * [Low-rank Adaption of Large Language Models Part 2: Simple Fine-tuning with LoRA](https://www.youtube.com/watch?v=iYr1xZn26R8)
  * [Low-rank Adaption of Large Language Models: Explaining the Key Concepts Behind LoRA](https://www.youtube.com/watch?v=dA-NhCtrrVE)
