# Source terms

Look up a word of the H-Net paper or the Multi-Gate Residuals (MGR) paper here
before you port from that paper. An entry gives our term and the wrong port that
the word can cause. Read the cited lines before you port from an entry.

`GLOSSARY.md` is the authority for our words, and this doc renames nothing
(ADR-0009). A word goes in the glossary if an agent that reads only our code and
docs needs it. A word goes here if only an agent that reads the sources needs it.
This doc has no word of the Mamba sources, because those words belong to
burn-mamba.

A source citation has the form `file:line`, with these path aliases. A citation
of our own code gives the item name.

| Alias | Path |
|---|---|
| `HN/` | `../papers/hnet/`: "Dynamic Chunking for End-to-End Hierarchical Sequence Modeling" |
| `HNM` | `../papers/hnet/sections/2_method_v2.tex` |
| `MGR` | `../papers/multi-gate-res/tex/main.tex`: "Multi-Gate Residuals" |

In the tables, **bold** marks a term of `GLOSSARY.md`.

## False friends and splits

A false friend is a source word that looks like a term of ours and means another
thing. A split is one source word that covers several terms of ours. The entries
go from the highest risk of a wrong port to the lowest. The order is a judgement.
A letter can also cause a wrong port, so also see [Letters](#letters).

| Source word | Its meaning in the source | Our term | Wrong port | Citation |
|---|---|---|---|---|
| "block", "Transformer block", "block depth", "residual block", "building block", "MLP blocks", "Block Size" | MGR: a whole Transformer block, with its attention layer and its feedforward layer. No line says this directly. The count `2·12−(4−1)=21` fits a block depth of 12 at `n=4` only with two source layers in each Transformer block. "Block Size" is the sequence length. H-Net: one layer, with two exceptions. "MLP blocks" is the feed-forward part. A related-work line uses "blocks" for chunks. | **Layer**. Not **Block**. **Feed-forward** for "MLP blocks". **Chunk** for the related-work line. The length of a **Sequence** for "Block Size". | A port that maps the word to our block can wrap a whole layer in a second Pre-LN and a second residual. | `MGR:171`, `MGR:268`, `MGR:318`, `MGR:339`, `MGR:424`, `MGR:433`, `MGR:684`, `HNM:67`, `HNM:228`, `HN/supplementary/supplementary.tex:7`, `HN/sections/A_background.tex:131` |
| "state", "DC state", "isotropic state" | H-Net: what each component keeps from one call to the next. The "isotropic state" is a "KV cache" for a Transformer layer and an "SSM state" for a Mamba-2 layer. The "DC state" is the last key of the router and the last `z̃` of the "dechunking layer". | **Cache**. In `HNetStageCache`, `router` keeps the last `x̂`, not the key, and `smooth_bd` keeps the last `z̄`. Only the "SSM state" is the State of burn-mamba. | A port that reads the word as the State of burn-mamba can store, pass or clear the wrong tensor. | `HNM:279`, `HNM:280`, `HNM:282`, `HNM:283` |
| "hidden state", `h_l`, "stream state" | MGR: `h_l` is the input of a layer, pooled from the streams. In a model with one stream, "hidden state" is that residual stream. "Stream state" is the residual streams. H-Net: the state of a recurrent model, or the vector that a stage passes to the "main network". | `h_l`: no term, the input `h` in the header of `multi_gate`. "Stream state", and "hidden state" of a model with one stream: **Residual stream**. H-Net: the State of burn-mamba, or no term. | A port can store, pass or clear `h_l` as the State of burn-mamba. | `MGR:105`, `MGR:141`, `MGR:155`, `MGR:171`, `MGR:193`, `MGR:343`, `MGR:432`, `MGR:468`, `HN/sections/4_discussion.tex:39`, `HN/supplementary/supplementary.tex:115`, `HN/supplementary/supplementary.tex:116`, `HN/fig/distill.tex:8` |
| "sequence mixing layer", `M`, `T` | H-Net: one layer. In the architecture table, `M` is a Mamba-2 layer and `T` is a Transformer layer. A Mamba layer has a gated MLP in the "main network", and none in an encoder or a decoder. | **Layer**. The gated MLP is its **Feed-forward**. | A port can add a feed-forward part to each Mamba layer of an encoder or a decoder. | `HNM:17`, `HN/sections/3_experiments.tex:50`, `HN/tab/architecture.tex:4` |
| "attention layer", "feedforward layer", `F_l` | MGR: two parts of one Transformer block. `F_l(h_l)` is the output of one such source layer. Streams merge from both parts, so one Transformer block has more than one source layer. No line says this directly. See the count in the "block" entry. | **Block** and **Feed-forward**, in one **Layer**. | A port can place a norm or a β gate in the wrong part of a layer. One `MultiGateResidual` runs for each layer of ours. | `MGR:171`, `MGR:193`, `MGR:197`, `MGR:268`, `MGR:433`, `MGR:467`, `MGR:487` |
| "RoPE" | MGR: the position embedding in the table of shared hyperparameters, with `θ=10000`. | No term in this crate. Not the Rotation of burn-mamba. | A port can read a fixed-frequency table as the data-dependent rotation of a family. | `MGR:677`, `MGR:695` |
| "gate" | MGR: `β`, the coefficient of one stream. The derivation of the bias init also uses "gate" for `g_l`, on one residual connection. | **β gate**. No term for `g_l`. Not the Gate of burn-mamba. | A port can read `β` as the SiLU gate `z`. | `MGR:172`, `MGR:193`, `MGR:237`, `MGR:240`, `MGR:523`, `MGR:544` |
| "stage", "main stage" | H-Net: an index `s`. Index `s=S` holds only the "main network". | **Stage** for `s<S`. **Main stack** for `s=S`. At `HNM:53`, "main stage" can be the **Inner H-Net**. | A port can build an encoder, a router and a decoder at index `s=S`. | `HNM:22`, `HNM:53`, `HN/supplementary/supplementary.tex:108` |
| "lerping stage" | MGR: the source layers from the `n`-th on, where the gates run. | No term: the mixing phase in the header of `multi_gate`. Not **Stage**. | A port can read the word as a stage of an H-Net. | `MGR:171`, `MGR:266`, `MGR:268`, `MGR:420` |
| "network", "network component", "sub-networks" | H-Net: a "network component" or a "sub-network" is a stack of layers, with an RMSNorm at its end. The bare word "network" can also be the whole H-Net. | **Layer stack**, with its RMSNorm outside it: `HNetStage::encoder_norm`, `HNetStage::decoder_norm`, `HNetMain::norm`. **H-Net** for the whole model. Not **Network**, which adds an input map and an output map. | A port can add an embedding and an LM head to each encoder and decoder. | `HNM:6`, `HNM:17`, `HNM:94`, `HNM:224`, `HNM:229`, `HN/sections/0_abstract.tex:7` |
| "main network", "inner stages" | H-Net: "main network" is one word for our main stack and our inner H-Net. "The main network can itself be an H-Net". "Inner stages" run on compressed representations. In a 2-stage H-Net, "the inner stage" also finds chunks. | **Main stack** or **Inner H-Net**. **Stage** for the "inner stage" that finds chunks. | A port can read a nested H-Net as one layer stack. | `HNM:17`, `HNM:19`, `HNM:22`, `HNM:75`, `HNM:256`, `HN/sections/1_introduction.tex:62`, `HN/sections/A_background.tex:164` |
| "residual connection", "residual structure", "additive residual update" | H-Net: the link from an encoder to its decoder, through a `Linear`. MGR: the residual scheme of a stack. | H-Net: `HNetStage::residual_proj`. MGR: **Residual scheme**. | A port can send the encoder-to-decoder link through the residual scheme of a layer stack. | `HNM:46`, `HNM:61`, `HNM:233`, `HNM:242`, `MGR:116`, `MGR:117`, `MGR:290`, `MGR:551` |
| "Separation of Two Streams" | H-Net: the two uses of an encoder output. One use is the link to its decoder, and the other is the input of the next stage. | No term. Not **Residual stream**. | A port can read the two uses as two residual streams of a layer stack. | `HNM:237`, `HNM:238` |
| "segmentation", "segment data" | H-Net: chunking. | **Chunk**, **Router**. Not **Segment**. | A port can link the word to the segments of a packed row. | `HN/sections/0_abstract.tex:6`, `HN/sections/1_introduction.tex:40` |
| "Kernel Fusion", "fuse core operators" | MGR: the fusion of operations into one kernel. | No term. Not **Fused projection**. | A port can read a kernel option as a fact about a fused projection, and so about the Muon plan. | `MGR:495`, `MGR:498`, `MGR:499` |
| "Tie Word Embedding" | MGR: a flag in the table of shared hyperparameters, with the value True. The source does not define it. Its name suggests embedding sharing. | No term. Not **Tied parameter**. | A port can read embedding sharing as sharing across applications. | `MGR:677`, `MGR:693` |
| "window", "width-k window" | H-Net: the attention range of Sliding Window Attention, or a pool width. | No term. Not **Window**. | A port can read an attention range as the part of a sequence that one forward takes. | `HN/tab/architecture.tex:8`, `HN/sections/A_background.tex:45` |
| "Straight-Through Estimator", `STE` | H-Net: it rounds a confidence to 1.0 and keeps its gradient. It adds no value-zero input at a cut. | No term: the `STE` in the header of `hnet`. Not **Straight-through carry**. | A port can read the estimator as the value-zero input at a cut. | `HNM:170`, `HNM:178`, `HNM:189` |
| "carry behavior" | MGR, from Highway Networks: the aim of the gate bias init. The source does not define the word. The init sets `β` small, so `1−β` is near 1 at the start of training. | The init of a **β gate**: `MultiGateResidualConfig::depth_init_bias`. Not **Straight-through carry**. | A port can read the word as the value-zero input at a cut. | `MGR:193`, `MGR:197`, `MGR:264`, `MGR:265`, `MGR:268` |
| "Fallback Inversion" | MGR: a strategy that saves GPU memory. The forward pass keeps the input streams at the largest gate values. The backward pass recovers the input streams by inversion, then reads the saved copy at the recorded indices. | No term. Not **Fallback optimizer**. | A port can read the word as the fallback optimizer. | `MGR:543`, `MGR:544`, `MGR:545`, `MGR:546`, `MGR:547` |
| "token", "BPE token", `Tokenizer` | H-Net: a footnote gives "one time step". Other method lines also use the word in this sense. Some lines, such as the abstract, mean a BPE token. For a hierarchical "byte-level" model, the `Tokenizer` column lists the chunking mechanism. For the baseline Transformer, it lists the GPT2 tokenizer. | **Token** for "one time step". No term for a BPE token. | A port can read a BPE token as one element of a sequence, or the chunking mechanism in the `Tokenizer` column as a tokenizer. | `HNM:53`, `HNM:271`, `HNM:285`, `HN/sections/0_abstract.tex:8`, `HN/tab/architecture.tex:5`, `HN/tab/architecture.tex:20` |
| "embeddings" | MGR: the only I/O weights that the source names for AdamW. | **I/O weight**, which also covers the LM head, the input and output projections and the class marker tables. | A port can leave the other I/O weights on Muon. The Muon plan lists no I/O weight. | `MGR:285`, `MGR:621` |

## Source terms with no term of ours

The third column gives the closest concept of ours. It is not the same concept.

| Source word | Meaning | Closest concept of ours | Citation |
|---|---|---|---|
| "chunking layer" | The routing module plus the downsampler. It is not a **Layer**. | **Router**, **Stage** | `HN/main.tex:97`, `HNM:105` |
| "dechunking layer" | The smoothing module plus the upsampler. It is not a **Layer**. | **Stage** | `HN/main.tex:100`, `HNM:130` |
| "downsampler" | It keeps the vectors with `b_t = 1`. It applies the same selection to `p^s`, and this gives `P^{s+1}`. | **Stage**, **Chunk**, the module `hnet::chunk` | `HNM:121`, `HNM:124`, `HNM:125` |
| "upsampler" | It repeats a vector until the next chunk boundary. Then it scales the vector by `STE(c_t)`. | **Stage**, **Chunk**, the module `hnet::chunk` | `HNM:158`, `HNM:178`, `HNM:191` |
| "smoothing module" | An EMA over chunks. Its weight `P_t` is the confidence of a chunk. | **Stage**, the function `smooth` | `HNM:136`, `HNM:139`, `HNM:147` |
| `p_t`, "boundary probability" | The score from cosine similarity. `p_1 = 1.0` forces a chunk boundary at the first position. | **Router**, **Chunk boundary** | `HNM:111`, `HNM:115`, `HNM:117` |
| `c_t`, "confidence score" | The confidence of the router in its chunk boundary decision. It is `p_t` if `b_t = 1`, and `1 - p_t` otherwise. `STE(c_t)` scales the upsampled vector, and its forward value is 1. | **Router** | `HNM:164`, `HNM:170`, `HNM:178`, `HNM:185`, `HNM:189` |
| "ratio loss" | A loss toward a target compression ratio `N`. | **Router**, `Routing::ratio_loss` | `HNM:201`, `HNM:204`, `HNM:206` |
| "dynamic chunking", "DC" | The chunking and dechunking mechanism. | **Router**, **Chunk** | `HNM:97`, `HNM:103`, `HNM:127`, `HNM:218` |
| "network normalization", "norm balance" | An RMSNorm at the end of each source network. | **Layer stack**, the norm fields of `HNetStage` and `HNetMain` | `HNM:226`, `HNM:229` |
| "learning rate modulation" | A learning-rate scale for each stage. | **LR schedule** | `HNM:253`, `HNM:258`, `HN/supplementary/supplementary.tex:39` |
| "isotropic" | A model with no hierarchy, or a component in that form. | **Layer stack**, **Main stack** | `HN/sections/1_introduction.tex:16`, `HNM:279` |
| "aggregator", "AttnPool", `α_{i→l}` | It pools the streams into the input of a layer. | **Residual stream**, **Multi-Gate Residuals (MGR)** | `MGR:148`, `MGR:149`, `MGR:150`, `MGR:154`, `MGR:155` |
| "independent", "competitive", `gate_variant` | The sigmoid form and the softmax form of the gate. Our `MultiGateResidual` has only the sigmoid form. The source class of that name has both, and its default is "competitive". | **β gate** | `MGR:172`, `MGR:177`, `MGR:572`, `MGR:574`, `MGR:577` |
| "gate bias", `b_{l,i}` | `b_{l,i}` is the bias of one stream. "Gate bias" also covers `b_l`, the bias of one gate in the derivation of the init. The softmax form sets `b_{l,0} = b_init` and `b_{l,i} = 0`. The sigmoid form sets `b_{l,i} = -b_init`. | **β gate** | `MGR:189`, `MGR:235`, `MGR:256`, `MGR:262`, `MGR:270`, `MGR:275` |
| "forget gate score", `b_{l,0}` | The extra entry of the softmax form. | **β gate** | `MGR:189`, `MGR:275` |
| "direct concatenation" | The source layers before the `n`-th append streams and do not gate. | **Residual stream**, `MultiGateResidual::accumulate` | `MGR:171`, `MGR:268` |
| "lerp" | The update of `s_i`. | **β gate**, `MultiGateResidual::forward` | `MGR:193`, `MGR:194`, `MGR:514` |

### Source names that a reader cannot decode

These words have a term of ours with the same concept. A row names each use that
has another concept.

| Source word | Meaning | Our term | Citation |
|---|---|---|---|
| "level", "multi-level" | "Multi-level" means a hierarchy with several stages. The stages model "levels of abstraction". "Byte level" and "character-level" name an input resolution, not a stage. | **Stage** for "multi-level". No term for "byte level". Not **Marker level**. | `HNM:7`, `HNM:19`, `HN/sections/0_abstract.tex:8`, `HN/sections/0_abstract.tex:9`, `HN/sections/0_abstract.tex:10` |
| "routing module", "chunking router", "boundary predictor" | It computes `p_t` and `b_t`. | **Router** | `HN/main.tex:109`, `HNM:53`, `HNM:105`, `HNM:111`, `HN/sections/1_introduction.tex:41`, `HN/supplementary/supplementary.tex:65` |
| `β_{i←l}`, "interpolation coefficient", "gating layer", "mixer", "residual stream mixers" | `β_{i←l}` and "interpolation coefficient" are the value of the gate of one stream, in a sigmoid form or a softmax form. "Gating layer" is the unit of one layer that computes this value for each stream. "Mixer" and "residual stream mixers" name that unit by its role: it mixes the layer output into the streams. That unit has one `w_l` for the layer and one bias for each stream. | **β gate** for one stream. A "gating layer" or a "mixer" holds the β gates of one layer. | `MGR:148`, `MGR:149`, `MGR:171`, `MGR:172`, `MGR:174`, `MGR:177`, `MGR:179`, `MGR:189`, `MGR:264`, `MGR:600` |
| "dual-parameter-group setup" | Muon updates "all 2D weight matrices". AdamW updates the rest: "biases, normalization layers, and embeddings". So embeddings do not go to Muon. Our plan is an allowlist with sub-projections. | **Muon plan** | `MGR:285`, `MGR:621` |
| AdamW for "all remaining scalar parameters" | The source has no noun for this role. It names only AdamW. "Scalar" does not mean rank 0: the examples include RMSNorm weights and linear layer biases, and `MGR:621` adds embeddings. | **Fallback optimizer** | `MGR:285`, `MGR:621` |

## Letters

A letter in the Notation table of `CLAUDE.md` can have another meaning in a
source. A port can bind the letter to the wrong axis or tensor. The port can
still pass the shape checks when two dimensions have equal sizes. The entries
that name burn-mamba use a letter of its Notation table.

| Our letter | Source use | Citation |
|---|---|---|
| `b` batch | `b_t` is the "boundary indicator" in H-Net. `b_t = 1` marks a position, but an intro line says "between adjacent elements". `b` is a gate bias in MGR. | `HNM:111`, `HNM:186`, `HN/sections/1_introduction.tex:41`, `MGR:184`, `MGR:189` |
| `s` sequence | `s` is the index of a stage and `S` is the count of stages in H-Net. `s_i` is a residual stream in MGR. | `HNM:22`, `MGR:193`, `MGR:233` |
| `d` d_model | `\mathcal{D}^s` is a decoder in H-Net. Plain `D^s` is the width of stage `s`, and this use agrees with our `d`. | `HN/main.tex:107`, `HNM:17`, `HNM:22`, `HNM:87`, `HN/sections/3_experiments.tex:50`, `HN/tab/architecture.tex:16` |
| `h` nheads | `h_l` is the input of a layer in MGR. | `MGR:155`, `MGR:193` |
| `p` per_head_dim | `p_t` is the "boundary probability" in H-Net. | `HNM:111`, `HNM:115` |
| `l` chunk_len | `L` is a sequence length in H-Net. `l` is the index of a layer in MGR, as in `F_l` and `h_l`. `L` is a count of layers in MGR. | `HNM:52`, `MGR:155`, `MGR:193`, `MGR:197`, `MGR:228`, `MGR:262`, `MGR:266` |
| `n` nchunks | `n` is the count of streams in MGR, and `N` is the same axis in its pseudocode. `N` is the target of the ratio loss in H-Net. | `MGR:273`, `MGR:595`, `HNM:204`, `HNM:206` |
| `k` conv kernel (burn-mamba) | `k_t` is a key in the H-Net router. | `HNM:115`, `HNM:282` |
| `m` MIMO rank (burn-mamba) | `M` is the "main network" in H-Net, and also a Mamba-2 layer in its architecture table. | `HNM:17`, `HN/tab/architecture.tex:4` |
| `β` in **β gate**, and in the Tap of burn-mamba | Two sources use one letter. `β_i` of MGR is the gate of a stream. The Tap takes its `β` from the Mamba-3 paper. | `MGR:193`, `MGR:544`, `../papers/mamba/mamba-3/structure/methods_trap.tex:138` |
