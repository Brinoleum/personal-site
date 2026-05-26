---
title: "ABCD (better title TBD)"
date: 2026-05-21T13:41:16-07:00
draft: true
---

## Intro

Genomic language models, large neural networks trained on raw DNA sequence rather than English text, have demonstrated that the same architectures powering modern NLP can be repurposed to operate on biological sequence data. Recent benchmarks show that these models can reliably distinguish between species from sequence alone, classifying human from worm or human from other primates with high accuracy. The natural extension of this capability is to classify *within* a species: given that all human samples share essentially the same genome at the species level, can a model be trained to recognize the variation between individuals that correlates with a particular phenotype?

This project investigates that question in the context of ADHD severity prediction. ADHD is a heritable condition, and standard GWAS analyses have identified many loci associated with its presentation; in addition, a growing body of work suggests that mitochondrial dysfunction is one of the underlying mechanistic threads. However, the majority of this evidence comes from SNP-by-SNP statistical analysis, which treats each locus independently. A foundation model, by contrast, processes the entire input sequence at once, and so in principle is capable of capturing combinations and contextual relationships between loci that single-locus methods cannot. Whether it does so in practice is the open question that motivates this work.

The remainder of this post is a writeup of the methods, since the model is still in training and results are not yet available. We describe the choice of a state-space model architecture over a transformer, the use of the Caduceus model in particular, our procedure for reconstructing per-subject genomes from PLINK files against the hg19 reference, and the data-augmentation problem that does not have a clean analogue when the input "language" is DNA.

## Background

### Genomic language modeling

Large language models such as BERT and the GPT family are commonly used for natural language processing tasks, where the input is a sequence of word or subword tokens and the model is trained to predict missing or following tokens given some context. Architecturally, these models are not specific to English text: the input is treated as an opaque sequence of token IDs, and the model learns the statistical relationships between tokens during pretraining on a large corpus. Any sequential data for which a sufficiently large corpus exists can in principle be modeled in the same way, since the architecture itself makes no assumption about what the tokens represent.

Genomic sequence data is one such input. The four nucleotide bases that comprise DNA are naturally analogous to characters in a written language, and the human reference genome together with the genomes of other organisms provides a corpus on the order of billions of bases. The pretraining objective is typically masked-language modeling, in the style of BERT: a fraction of the input bases are hidden from the model, and it is trained to recover them from the surrounding context. There are now several families of genomic language models — DNA-BERT, DNA-GPT, HyenaDNA, and Caduceus among others — that apply this recipe with various architectural choices, and most of them are pretrained primarily on the human reference genome. Fine-tuning on a task-specific dataset, as we do in this project, is therefore the standard way to specialize one of these pretrained models to a particular downstream problem.

One design choice worth mentioning at this stage is tokenization. In natural language, tokens are typically words or subword units learned from the training corpus, since operating at the character level would force the model to learn word-level structure from scratch. For DNA, the analogous choice is between single-character tokenization (one token per nucleotide) and k-mer tokenization (one token per fixed-length window of bases). Triplet tokenization is a particularly tempting option, since the cellular machinery itself reads DNA in groups of three when translating to amino acids during protein synthesis. However, k-mer tokenization is fragile to insertions and deletions: adding or removing a single base shifts the reading frame, so every subsequent token in the sequence changes even though the underlying biology has only changed in one place. Single-character tokenization avoids this problem at the cost of longer input sequences, and is the choice we make later for our specific task.

The capabilities of these models are evaluated on benchmarks adapted from the same principle as their NLP counterparts. The Nucleotide Transformer benchmark suite is the most commonly cited collection, and includes a range of tasks beyond the species-classification example mentioned above: promoter prediction, where the model is given a sequence and asked whether it contains a transcription-initiation site; splice-site prediction, where it is asked to identify the boundaries between coding and non-coding regions of a gene; and variant effect prediction, where the model is asked to estimate whether a given mutation is likely to be functionally consequential. The last of these is particularly relevant to the present work, since predicting ADHD severity from sequence is in essence a variant-effect problem at the level of an entire phenotype rather than a single locus: we are asking the model to integrate the effects of many variants across the genome and produce a single behavioral readout. Results across the Nucleotide Transformer tasks show that genomic language models are picking up on real biological structure rather than surface-level sequence statistics, which is what makes them a plausible starting point for the fine-tuning task described in the rest of this post.

### ABCD data

The Adolescent Brain Cognitive Development study is a long-term longitudinal study that tracks brain development in roughly 12,000 subjects between the ages of 9 and 20. The dataset is multimodal, collecting fMRI scans, behavioral and diagnostic questionnaires, and cheek swabs from which DNA is sampled. Of these, we focus on two specific data points: the gene sequence derived from each subject's DNA sample, and the subject's KSADS questionnaire score, which provides a clinician-administered diagnostic measurement on a 0-to-4 severity scale.

## Hypothesis

Mental health conditions such as ADHD are known to be heritable, and SNP-based statistical analyses have identified specific mutations whose presence correlates with higher rates of diagnosis. The hypothesis driving this project is that a deep learning model trained on the full genome should be able to recover similar signal, and ideally more, since it has access to the full sequence rather than individual loci considered in isolation.

In addition to the question of whether such a model classifies correctly, there is a follow-up question of mechanistic interpretability: when the model makes a classification decision, what parts of the input is it actually attending to? If the regions it identifies correspond to genes or regulatory elements already known to be involved in the condition, that provides some validation of the approach; if the model attends to regions that are not currently part of the biological picture, those regions become candidates for further investigation.

### Biological basis

The specific biological mechanism we are interested in is mitochondrial dysfunction. The current theory holds that reduced energy production in neurons impairs the ability of brain cells to self-regulate, and that the regions of the brain responsible for sustained attention are particularly sensitive to this effect. Accordingly, our region of interest within the genome is focused on mtDNA, as well as on the genes and regulatory regions on the nuclear chromosomes that are responsible for mitochondrial function.

## Implementation

### Model selection

The size of a single human genome (on the order of 3 billion base pairs) makes the standard transformer architecture poorly suited as a starting point: the O(n²) time complexity of self-attention limits practical input windows to the order of 10,000 to 100,000 tokens, which is several orders of magnitude shorter than the sequences of interest. State-space model (SSM) architectures, by contrast, replace self-attention with a long-convolution and sliding-window mechanism that runs in subquadratic time, and recent work has demonstrated input windows of up to 1 million tokens with these architectures.

Two foundation models in this family are publicly available and pretrained on genomic data: HyenaDNA and Caduceus. We selected Caduceus for this project, primarily for the reverse-complement equivariance built into its architecture, which matches the underlying biology of double-stranded DNA.

### Model setup

We follow the recommended setup from the Caduceus authors for the most part. In particular, we use single-character tokenization rather than k-mer tokenization, since our task operates at the level of single-nucleotide polymorphisms: a single-base insertion or deletion has an outsized effect on downstream sequence in a way that the model needs to be sensitive to. Tokenizing in triples, analogous to the codon structure used during translation, would obscure these shifts by changing every subsequent token after the variant. The pretrained head of the model is then replaced with a classification head sized for our task, with output dimension equal to the five-point KSADS severity scale.

### Data preprocessing

The ABCD dataset does not distribute per-subject genome sequences directly. Instead, each subject's genotype is provided as a PLINK file, which specifies, for each SNP, the nucleotide substitution (for example, A to G) and its position in terms of chromosome and base-pair offset, calculated relative to the hg19 reference genome. In order to recover the per-subject sequence that serves as input to the model, we apply each variant in the PLINK file to the corresponding position in hg19, and repeat the process for every subject in the dataset.

### Training loop

The training loop itself is a fairly standard classification setup: the data is partitioned into training and test sets, the model is trained on the training set, and accuracy on the held-out test set is used to validate the results.

## Learnings

### Data augmentation

The primary practical difficulty encountered so far has been data augmentation. The ABCD dataset, while large by clinical standards, is small relative to the scale of the model, and the class distribution is imbalanced across KSADS severity levels. Both issues point toward the need for synthetic data augmentation, but the techniques that work well for natural language do not transfer cleanly to genomic data. In natural language, common augmentation strategies include replacing words with synonyms or rearranging clauses, both of which approximately preserve the meaning of the sentence. The genomic analogue of these operations — substituting nucleotides at random or shuffling subsequences around — corresponds to large-scale mutations that may not be compatible with life, and so cannot be assumed to preserve the phenotype label associated with the original sample. Our current approach is therefore to forgo synthetic augmentation altogether and simply resample the existing data to balance the class distribution.

## Next steps

Getting the model to classify correctly is one part of the goal; understanding *why* it classifies the way it does is the other. The latter is a question of mechanistic interpretability, and concerns what the model attends to when making a decision. A model that classifies correctly without yielding interpretable internals is of limited scientific use in this context, since the value of this approach over traditional GWAS lies precisely in the ability to surface combinations and contexts that single-locus methods miss. The next phase of the project, once training has converged, is to apply interpretability tools to the trained model and to compare the regions it attends to against the current biological understanding of ADHD and mitochondrial function.

#### Citations
TODO: collate Zotero list
