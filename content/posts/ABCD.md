---
title: "ABCD (better title TBD)"
date: 2026-05-21T13:41:16-07:00
draft: true
---

## Intro

Can a foundation model trained on raw DNA predict how severely a teenager will present with ADHD — and if so, what is it actually looking at?

That's the question driving this project. Genomic language models have shown they can tell species apart from sequence alone — human vs. worm, human vs. other primates — which is really just classification on sequence data. The leap I'm interested in is whether the same machinery can classify *within* a species: same genome at the species level, different phenotypes at the individual level. ADHD is a reasonable target. It's heritable, GWAS has surfaced plenty of associated loci, and there's a growing body of work pointing at mitochondrial dysfunction as one of the mechanistic threads. Most of that evidence comes from SNP-by-SNP statistical analysis; a foundation model gets to see the whole sequence at once, which in principle lets it pick up on combinations and context that single-locus methods miss. Whether it actually does is the open question.

A note on scope: this is a writeup of the methods, not the results. The model is still training. What follows is the setup — why a state-space model over a transformer, why Caduceus specifically, how I rebuilt per-subject genomes from PLINK files against hg19, and the data-augmentation problem that doesn't have a clean answer when your "language" is DNA.

## Background

### Genomic language modeling
- Transformers and related model architectures are well suited for sequential data, not just English
- Genome data is also inherently sequential, and there exist many models (DNA-GPT, DNA-BERT, etc) using the same fundamental bones, but trained on DNA data
- Benchmarks show good results with differentiating between different species (human vs worm, human vs primate)

### ABCD data
- Adolescent Brain Cognitive Development study: long term, longitudinal study tracking brain development in teens ages 9-20
- Multimodal data collection: fMRI, behavioral diagnostic questionnaires, cheek swabs to sample DNA
- 12,000 study subjects

## Hypothesis
- mental disorders like ADHD, etc are correlated to genetic causes
- SNP analysis shows higher occurrences of certain diagnoses are tied to specific mutations
- proposed analysis with deep learning
    - what kinds of insights can we get out of training a model to detect illness?
    - mechanistic interpretability -> biological outcomes: what is the model looking at when it does its classification work? does that correlate with current biological understanding?

### Biological basis
- ADHD in particular is correlated to mitochondrial dysfunction; current theory is that the lack of energy production means that brain cells are less able to self-regulate, and areas responsible for attention are particularly affected
- our focus is on the genes, regulatory/functional areas responsible for mitochondrial function in particular
    - mtDNA, other chromosomes

## Implementation

### Model selection
- experimentation with model architecture: state space modeling
    - transformers are suboptimal for our use case: O(n^2) time complexity, order of 10-100k token input window
    - SSM architectures are an active field of research: Mamba modules
    - long convolution/sliding window approach yields up to 1M input tokens in subquadratic time
- ready availability of models:
    - HyenaDNA, Caduceus are two examples of foundational models, pretrained on genomic data, that we can leverage for our own classification tasks
- we worked with caduceus for this project

### Model setup
- following mostly the recommended setup
    - single character tokenization since we work on the level of SNPs so mutations like a single deletion have outsize effects: if we tokenize three at a time (almost like the transcription process for amino acids) or more, then the remainder of the sequence gets changed as well and that may throw off the model
- replace the pretrained head of the model with our own classification head:
    - more detail in the data processing section, but we want to classify by severity of diagnosis, so we need to ensure that the model is configured for that goal

### Data preprocessing
- multimodal dataset has many points collected per sample but we focus primarily on two things in particular: gene sequence and KSADS questionnaire score
- gene sequence is the input to the model and the output is the projected severity of the diagnosis (KSADS, scale 0 to 4)
- recreate the gene sequence from scratch:
    - dataset comprised of PLINK files describing the specific SNP/mutation (for example 'A' to 'G') and its location (chromosome, base pair) calculated relative to reference genome hg19
    - so we go in reverse: go to the reference genome at each location described in the PLINK file and apply the SNP
    - do this for all samples to generate the dataset


### Training loop
- fairly standard classification setup
- divide data into training and test sets, train model, validate results

## Learnings

### Data augmentation
- due to combination of imbalanced data, not enough samples for the scale of the model:
    - need to have some way to synthetically generate new data to train the model on
    - techniques for natural language don't translate well to genomics:
        - you can replace words with synonyms, rephrase and move clauses around with natural language
        - shuffling around data for the genome is tantamount to larger scale mutations that may or may not be compatible with life
- for now, solution is to simply resample the data to get roughly even balance of data

## Next steps
- getting a model to classify correctly is one thing, understanding why it does so is another
- this is really more a question of mechanistic interpretability:
    - what the model looks at when it makes these decisions
    - positive results mean that we have something interesting going on
    - need to know what it's looking at that we might have otherwise missed

#### Citations
TODO: collate Zotero list