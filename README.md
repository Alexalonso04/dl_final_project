# GPT Optimizations

## Summary
‘Recent studies have shown that LLMs face challenges in accurately retrieving key information from context’ which causes degraded performance and confabulations. One reason for this is that traditional Transformer architecture tends to allocate significant attention to irrelevant context, which can drown out important information, particularly as context lengths increase. 
This work will begin an exploration of recent methods that attempt to deal with this problem. In particular, we will begin by focusing on and implementing the Differential Transformer [1], a method that demonstrates the ability to cancel out attention noise by calculating the attention as the difference between two separate SoftMax attention maps. This approach amplifies attention to relevant context while canceling out noise, promoting more focused and efficient information processing. 
Empirical results show that this architecture demonstrates improved performance  in terms of scaling properties, long-context modeling, key information retrieval, hallucination(confabulation) mitigation, in-context learning, and reduction of activation outliers. We believe that this research direction will provide valuable insights into building more efficient and reliable language models. 

## Approach
We will first test this architecture by implementing the differential transformer in nanoGPT[2] using the fineweb-edu 10B dataset [3].
We plan to use nanoGPT, and the results from the subsequent modded-nanoGPT community, to get a sense of how much the differential transformer actually improves performance. Additionally, we will evaluate it on the same set of Eval Harness[10] tasks used in the differential transformer paper.
After performing the initial benchmarks, we will start experimenting with finetuning.  We will then run our benchmarks again, and compare them with the baseline version.
If time allows, we also believe that it would be worth exploring relatively simple augmentations, such as selective attention [8], to see if the attention noise cancellation effects stack as it seems straightforward to implement (efficient implementation notwithstanding).

## Resources
The rapid pace of research in large language models makes it challenging to consistently define the state of the art. Starting with transformers, active research areas include differential transformers[1], which work to reduce attention noise, selective attention mechanisms[8], which allow models to prioritize critical parts of the input, and normalized transformers[5], which apply hyperspherical normalization to improve representation stability. Lastly, some recent work even involves dropping entire attention layers [9]. All of these techniques help mitigate hallucination/confabulations in LLMs.  
For benchmarking large language models, a notable method includes HellaSwag [10], which tests a model’s ability to understand context in diverse situations.
Overall, these approaches represent a sample of state-of-the-art methods that we aim to incorporate into our implementation.

### Resource Papers:
[1] Differential Transformer, https://arxiv.org/abs/2410.05258
[2] Nano GPT:  https://github.com/karpathy/build-nanogpt
[3] FineWeb-Edu Database: https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/tree/main/sample/10BT
[4] Modded NanoGPT: https://github.com/KellerJordan/modded-nanogpt
[5] nGPT: Normalized Transformer with Representation Learning on the Hypersphere, https://arxiv.org/abs/2410.01131
[6] Towards Small, Faster Decoder-Only Transformers: Architectural Variants and Their Implications, https://arxiv.org/abs/2404.14462
[7]The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits, https://arxiv.org/abs/2402.17764
[8] Selective Attention Improves Transformers, https://arxiv.org/abs/2410.02703
 [9] What Matters In Transformers? Not All Attention is Needed, https://arxiv.org/abs/2406.15786
[10]Eval Harness https://github.com/EleutherAI/lm-evaluation-harness

## Datasets
FineWeb Edu 10B token Dataset: https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/tree/main/sample/10BT

Fine Tuning:
 Conversational Question Answering, https://stanfordnlp.github.io/coqa/
CommonsenseQA, https://huggingface.co/datasets/tau/commonsense_qa
