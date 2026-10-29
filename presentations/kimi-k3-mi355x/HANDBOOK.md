# Kimi-K3 on AMD MI355X — handbook

This is a conceptual course with an editable animated PowerPoint, a static reference PowerPoint/PDF, a narrated chaptered video and detailed Markdown notes.

## How to use it

- Open **Kimi-K3_MI355X_Animated.pptx** in desktop PowerPoint and start Slide Show (F5). Click to reveal the next operation. The Animation Pane exposes editable fade entrance groups.
- The **Reference** PowerPoint and PDF show complete diagrams immediately.
- Each slide's **READ CHAPTER** link opens the relevant online Markdown section. The same files are included here for offline reading.
- The video uses the same progressive diagram states, English synthetic narration and captions. Video and PowerPoint timings are designed separately: the deck is presenter-controlled, the video is narrated.
- Green = FP8; blue = BF16; purple = FP32; amber = conversion/decision. Color is always accompanied by text.
- Model dimensions are pinned to the linked checkpoint. Diagrams of execution and hardware topology are schematic unless explicitly labeled otherwise.

## Learning routes

**Start from basics:** chapters 01–03, then the attention and KDA worked examples.

**Understand the GPU:** chapters 04–05, then matrix layout, state sizing and graph-replay notes.

**Understand the FP8 experiment:** chapters 06–08 plus quantization units, LSE merging and accuracy examples.

**Complete workshop:** all slides in order; pause at the six understanding questions. Allow multiple sessions for the full material.

## Chapters

- [00. Orientation](guide/00-orientation.md)
- [01. Attention fundamentals](guide/01-attention-fundamentals.md)
- [02. Fused attention](guide/02-fused-attention.md)
- [03. Kimi-K3 architecture](guide/03-kimi-k3-architecture.md)
- [04. MI355X execution](guide/04-mi355x-execution.md)
- [05. Distributed serving](guide/05-distributed-serving.md)
- [06. FP8 data flow](guide/06-fp8-data-flow.md)
- [07. Accuracy and execution proof](guide/07-accuracy-and-execution-proof.md)
- [08. Performance reasoning](guide/08-performance-reasoning.md)
- [09. Putting it together](guide/09-putting-it-together.md)
- [10. Reference](guide/10-reference.md)

- [11. Worked examples, math, sampling and common misunderstandings](guide/11-worked-examples.md)
- [Sources and evidence](SOURCES.md)

## Important implementation distinction

This checkpoint sets mla_use_nope=true. The 64-column field named qk_rope_head_dim is retained in the generic MLA layout, but the inspected K3 MLA wrapper passes rotary_emb=None. The diagrams call it the shared K component rather than claiming rotary position encoding is active.
