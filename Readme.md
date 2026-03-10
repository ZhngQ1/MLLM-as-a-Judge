# Vision-Judge: Evaluating Multi-modal Large Language Models as MLLM-as-Judge for Vision Tasks

> **Paper status:** Manuscript in preparation. Code and data will be released upon publication.

## Overview

While multi-modal large language models (MLLMs) have shown remarkable capabilities, they remain **unreliable and biased** when used as judges for vision-centric tasks — especially in **tool-mediated agentic settings** where models must interpret structured outputs from vision tools (e.g., detectors, segmenters, OCR systems) to make downstream decisions.

**Vision-Judge** is a benchmark that systematically evaluates MLLMs as judges of vision-task outputs along two axes:

- **Intrinsic Judging**: Can the MLLM assess vision-task results directly from the input image and candidate outputs?
- **Tool-Mediated Judging**: Can the MLLM correctly interpret and leverage outputs produced by vision tools to select the better result?

## Key Findings

- State-of-the-art MLLMs are competitive on straightforward intrinsic judging, but **performance drops substantially** in tool-mediated settings.
- Models exhibit **brittle parsing of tool outputs**, **poor calibration**, and **sensitivity to formatting** of tool results.
- There is a **systematic bias toward verbose or high-confidence candidates**, even when contradicted by tool evidence.

## Benchmark Design

The benchmark covers multiple vision tasks and judging formats, with emphasis on:

- **Pairwise preference judgments** with rationale-grounded decisions
- **Challenging cases** where candidates trade off correctness, completeness, and visual fidelity
- **Diagnostic tags** capturing common failure modes, including:
  - Over-reliance on irrelevant tool fields
  - Misinterpretation of numeric scores
  - Conflating perceptual plausibility with task correctness

## Tasks Covered

This repository focuses on the **image generation** track of Vision-Judge, including:

| Task | Description |
|------|-------------|
| Text-to-Image | Evaluating generated images against text prompts |
| Inpainting | Assessing quality of filled-in image regions |
| Image Editing | Judging accuracy of instruction-based edits |
| Conditional Generation | Evaluating outputs conditioned on structural inputs |

For instance segmentation evaluation, see the companion repository: [Instance-Segmentation-for-MLLM-as-a-Judge](https://github.com/ZhngQ1/Instance-Segmentation-for-MLLM-as-a-Judge).

## Repository Structure

```
├── Model/            # Model inference and evaluation scripts
├── Results/          # Evaluation results and analysis
├── main.ipynb        # Main evaluation pipeline
└── README.md
```

## My Contributions

- Co-designed the benchmark framework and evaluation protocol
- Curated datasets for multiple vision tasks with pairwise preference annotations
- Designed diagnostic failure-mode tags for systematic error analysis
- Contributed to paper writing and experimental analysis

## Citation

```
Paper citation will be added upon publication.
```

## Acknowledgments

This work was conducted at the [Statistical Visual Computing Laboratory (SVCL)](http://www.svcl.ucsd.edu/), UC San Diego, under the supervision of Prof. Nuno Vasconcelos.
