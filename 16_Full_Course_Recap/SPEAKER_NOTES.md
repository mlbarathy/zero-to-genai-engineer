# Speaker notes · Full-course recap

Students open:
https://nursnaaz.github.io/zero-to-genai-engineer/16_Full_Course_Recap/RECAP_S00_S15_INTERACTIVE.html

S00 to S07 interview reveals come from the Session 08 question bank in `RECAP_SLIDES.html`. S09 to S15 questions follow the same style, drawn from the session READMEs and teaching decks.

The deck speaks to the students directly, so you can also hand them the link and let them run it alone.

## Room habits

1. Ask the question. Wait. Let the silence do the work.
2. Click **Show a strong answer**. Compare it with what the room said.
3. Run every **Play** before you talk over it.
4. Ship links are GitHub or Pages only. Open them on the projector.

The score counter in the top left tracks first attempts on the 14 trap questions, so a wrong answer sticks. Tell them that up front.

Do not replace Session 08. The S08 stage is a bridge with links to the mid-course decks.

## New in this version

- `playGraph` used to light the GPT-scale boxes instead of the LangGraph nodes. Fixed, and the retry edge now animates properly.
- S09 teaches your real `/goal` formula (stop condition as an exact shell command, plus the small judge model reading the transcript each turn) rather than a generic five-box loop.
- S15 now carries the memory arithmetic from your fine-tuning deck: 112 GB full fine-tune, 15 GB free Colab, 14 / 3.5 / 0.05 / 3.55 GB, and 8,192 against 262,144 at rank 8.
- New stages: S03 alignment (RLHF against DPO), S10 reranking, S11 human in the loop, S04 and S15 interview splits, S15 memory math.
- Two duplicated S04 interview cards removed.

## Chapter checks (can they say this?)

| Chapter | Must be able to say |
|---|---|
| S00 | Inverted index. IDF = 0 when the term is everywhere. TF-IDF cannot do synonyms. |
| S01 | Cosine = angle. Word2Vec is static. Transformers fix context. |
| S02 | Q, K, V. Why sqrt(d_k). Why positional encoding. |
| S03 | In-context learning. RLHF steps. DPO drops the reward model. |
| S04 | BPE merges. Temp then top-k then top-p then sample. |
| S05 | GGUF / Q4. Local vs OpenRouter tradeoffs. Universal chat roles. |
| S06 | DSPy signature + metric. Bootstrap vs MIPROv2. |
| S07 | LCEL pipe. Stateless API. Streaming. Swap providers in one line. |
| S09 | /goal has a finish line, /loop does not. The stop condition is a shell command with exact output. |
| S10 | Hybrid when exact tokens matter. RRF fuses ranks. Rerank so the winner is not buried. Faithfulness. Memory is not RAG. |
| S11 | Graphs for branch and HITL. Supervisor routes specialists. interrupt plus thread_id makes the pause durable. |
| S12 | Tool vs skill. AGENT.md stands always. |
| S13 | RAG path is not SQL path. HITL before menu write. |
| S14 | No AWS keys in the browser. us-east-1 vs us-west-2. |
| S15 | RAG facts. LoRA behavior. Quantization is not new knowledge. |

## WhatsApp polls

- Before hybrid: query is `ERR-4042`. Dense only or hybrid?
- Before Dining Bot: can the LLM run UPDATE SQL?
- Before S15 table: 14-day refund. RAG or fine-tune?

## Wrong answers you will hear

- IDF is huge for common words → write log(N/N)=0
- Euclidean is fine for reviews → short vs long same meaning
- Sample before temperature → call it a production bug
- The API remembers chats → stateless
- Quantization teaches the PDF → bits only
