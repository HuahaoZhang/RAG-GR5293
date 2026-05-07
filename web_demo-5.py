"""
web_demo.py - No-tab version to avoid gradio tab freeze issues
"""
import os, sys, time, json
from dotenv import load_dotenv
load_dotenv()

if not os.environ.get("OPENAI_API_KEY"):
    print("ERROR: OPENAI_API_KEY not set.")
    sys.exit(1)

import gradio as gr

_pipelines = {}
_ready = False

def get_pipelines():
    global _pipelines, _ready
    if not _ready:
        from src.retriever import DenseRetriever
        from src.generator import Generator
        from src.pipeline import BaselinePipeline, V1Pipeline, V2Pipeline, V3Pipeline
        retriever = DenseRetriever()
        retriever.index_chunks()
        generator = Generator()
        _pipelines = {
            "baseline": BaselinePipeline(retriever, generator, top_k=5),
            "v1": V1Pipeline(retriever, generator, top_k=5),
            "v2": V2Pipeline(retriever, generator),
            "v3": V3Pipeline(retriever, generator),
        }
        _ready = True
    return _pipelines

def run_single_query(question, variant, top_k):
    if not question.strip():
        return "Please enter a question.", "", ""
    p = get_pipelines()
    key = variant.lower().split()[0]
    r = p["v2"].run(question, top_k=int(top_k)) if key == "v2" else p.get(key, p["baseline"]).run(question)
    cache_str = "HIT" if r.get("cache_hit") else "miss"
    complexity_str = ""
    if "complexity_score" in r:
        complexity_str = f"\n| Complexity | {r['complexity_score']:.3f} — {r.get('complexity_tier','')} |"
    stats = f"| Metric | Value |\n|--------|-------|\n| Cache | {cache_str} |\n| top-k | {r['top_k_used']} |\n| Retrieval | {r['retrieval_latency_ms']:.0f} ms |\n| Generation | {r['generation_latency_ms']:.0f} ms |\n| **Total** | **{r['total_latency_ms']:.0f} ms** |{complexity_str}"
    chunks = "".join([f"**[{i+1}] {c.get('title','')[:50]}** sim:{c.get('score',0):.3f}\n\n>{c['text'][:200]}...\n\n---\n\n" for i,c in enumerate(r["chunks"][:5])])
    return r["answer"], stats, chunks

def run_cache_demo(question):
    if not question.strip():
        return "Please enter a question."
    p = get_pipelines()
    p["v1"].cache._store.clear()
    p["v1"].cache.reset_stats()
    t0 = time.perf_counter(); r1 = p["v1"].run(question); cold = (time.perf_counter()-t0)*1000
    t1 = time.perf_counter(); r2 = p["v1"].run(question); warm = (time.perf_counter()-t1)*1000
    hit = r2.get("cache_hit", False)
    speedup = cold / max(warm, 0.001)
    return f"### Cold pass\n- Latency: **{cold:.0f} ms** — full retrieval + generation\n- Cache: miss\n\n### Warm pass\n- Latency: **{warm:.2f} ms**\n- Cache: {'HIT' if hit else 'miss'}\n{'- **'+f'{speedup:,.0f}x faster**' if hit else ''}\n\n### Answer\n{r1['answer']}"

def run_topk_sweep(question):
    if not question.strip():
        return "Please enter a question."
    p = get_pipelines()
    rows = "| top-k | Latency | Answer |\n|-------|---------|--------|\n"
    for k in [1, 3, 5, 10, 20]:
        r = p["v2"].run(question, top_k=k)
        mark = " SWEET SPOT" if k == 5 else ""
        rows += f"| k={k}{mark} | {r['total_latency_ms']:.0f} ms | {r['answer'][:60]} |\n"
    return rows

def run_easy(q):
    p = get_pipelines()
    r = p["v3"].run(q)
    stats = f"| Metric | Value |\n|--------|-------|\n| Complexity | {r.get('complexity_score',0):.3f} — {r.get('complexity_tier','')} |\n| top-k | {r['top_k_used']} |\n| Latency | {r['total_latency_ms']:.0f} ms |"
    return r["answer"], stats

def run_hard(q):
    p = get_pipelines()
    r = p["v3"].run(q)
    stats = f"| Metric | Value |\n|--------|-------|\n| Complexity | {r.get('complexity_score',0):.3f} — {r.get('complexity_tier','')} |\n| top-k | {r['top_k_used']} |\n| Latency | {r['total_latency_ms']:.0f} ms |"
    return r["answer"], stats

def show_results():
    path = "results/summaries.json"
    if not os.path.exists(path):
        return "No results found.", ""
    with open(path) as f:
        s = json.load(f)
    main_md = "| Variant | F1 | EM | Mean Latency | P50 | Cache Hit |\n|---------|----|----|-------------|-----|-----------|\n"
    for name in ["baseline","v1_exact_cache","v3_adaptive"]:
        if name in s:
            r = s[name]
            main_md += f"| {name} | {r['f1']:.4f} | {r['exact_match']:.4f} | {r['mean_total_latency_ms']:.0f}ms | {r['p50_total_latency_ms']:.0f}ms | {r['cache_hit_rate']:.2%} |\n"
    sweep_md = ""
    if "v2_topk_sweep" in s:
        sweep_md = "\n### V2 top-k Sweep\n\n| top-k | F1 | Mean Latency | P50 |\n|-------|----|----|-----|\n"
        for k in [1,3,5,10,20]:
            key = f"topk_{k}"
            if key in s["v2_topk_sweep"]:
                r = s["v2_topk_sweep"][key]
                mark = " SWEET SPOT" if k==5 else ""
                sweep_md += f"| k={k}{mark} | {r['f1']:.4f} | {r['mean_total_latency_ms']:.0f}ms | {r['p50_total_latency_ms']:.0f}ms |\n"
    return main_md, sweep_md

SAMPLE_QS = ["Who directed the movie Inception?","What is the capital of France?","What year did World War II end?","Which country invented the telephone?","What is the speed of light?"]
COMPLEX_QS = ["Which author was born earlier, the writer of The Hobbit or the creator of A Song of Ice and Fire?","Which magazine was started first, Fortune or Time?"]

def build_ui():
    with gr.Blocks(
        title="RAG System Demo",
        theme=gr.themes.Soft(
            primary_hue="teal",
            neutral_hue="slate",
            font=[gr.themes.GoogleFont("Noto Sans"), "Arial", "sans-serif"],
        ),
    ) as demo:
        gr.Markdown("# RAG System Demo\n### Optimizing Caching and Retrieval Strategies for Latency-Quality Trade-offs")

        feature = gr.Radio(choices=["Single Query","Cache Effect","top-k Sweep","Adaptive Retrieval (V3)","Experiment Results"], value="Single Query", label="Select Feature")

        with gr.Group(visible=True) as grp_single:
            gr.Markdown("### Single Query — run any variant")
            with gr.Row():
                with gr.Column(scale=3):
                    q1 = gr.Textbox(label="Question", placeholder="Ask anything...", lines=2)
                    with gr.Row():
                        v1_dd = gr.Dropdown(choices=["baseline","v1","v2","v3"], value="baseline", label="Variant")
                        k1 = gr.Slider(minimum=1, maximum=20, step=1, value=5, label="top-k (V2 only)")
                    btn1 = gr.Button("Run", variant="primary")
                    gr.Examples(examples=[[q] for q in SAMPLE_QS], inputs=q1, label="Sample questions")
                with gr.Column(scale=2):
                    ans1 = gr.Textbox(label="Answer", lines=4)
                    stats1 = gr.Markdown()
            chunks1 = gr.Markdown()
            btn1.click(show_progress="minimal", fn=run_single_query, inputs=[q1,v1_dd,k1], outputs=[ans1,stats1,chunks1])

        with gr.Group(visible=False) as grp_cache:
            gr.Markdown("### Cache Effect — same question twice")
            cq = gr.Textbox(label="Question", value=SAMPLE_QS[0], lines=1)
            cb = gr.Button("Run Cache Demo", variant="primary")
            co = gr.Markdown()
            cb.click(show_progress="minimal", fn=run_cache_demo, inputs=cq, outputs=co)

        with gr.Group(visible=False) as grp_sweep:
            gr.Markdown("### top-k Sweep — k = 1, 3, 5, 10, 20")
            sq = gr.Textbox(label="Question", value=SAMPLE_QS[0], lines=1)
            sb = gr.Button("Run Sweep", variant="primary")
            so = gr.Markdown()
            sb.click(show_progress="minimal", fn=run_topk_sweep, inputs=sq, outputs=so)

        with gr.Group(visible=False) as grp_adaptive:
            gr.Markdown("### Adaptive Retrieval (V3) — simple vs complex")
            with gr.Row():
                with gr.Column():
                    gr.Markdown("**Simple question**")
                    aq_e = gr.Textbox(value=SAMPLE_QS[0], label="Question", lines=1)
                    ab_e = gr.Button("Run", variant="primary")
                    aa_e = gr.Textbox(label="Answer", lines=2)
                    as_e = gr.Markdown()
                with gr.Column():
                    gr.Markdown("**Complex question**")
                    aq_h = gr.Textbox(value=COMPLEX_QS[0], label="Question", lines=2)
                    ab_h = gr.Button("Run", variant="primary")
                    aa_h = gr.Textbox(label="Answer", lines=2)
                    as_h = gr.Markdown()
            ab_e.click(show_progress="minimal", fn=run_easy, inputs=aq_e, outputs=[aa_e, as_e])
            ab_h.click(show_progress="minimal", fn=run_hard, inputs=aq_h, outputs=[aa_h, as_h])

        with gr.Group(visible=False) as grp_results:
            gr.Markdown("### Experiment Results (n=200)")
            res_main = gr.Markdown()
            res_sweep = gr.Markdown()
            load_btn = gr.Button("Load Results", variant="primary")
            load_btn.click(show_progress="minimal", fn=show_results, outputs=[res_main, res_sweep])

        groups = [grp_single, grp_cache, grp_sweep, grp_adaptive, grp_results]
        labels = ["Single Query","Cache Effect","top-k Sweep","Adaptive Retrieval (V3)","Experiment Results"]

        def switch(choice):
            return [gr.update(visible=(choice == lbl)) for lbl in labels]

        feature.change(fn=switch, inputs=feature, outputs=groups)

    return demo

if __name__ == "__main__":
    print("Loading RAG system...")
    get_pipelines()
    print("System ready.")
    ui = build_ui()
    ui.queue(max_size=5)
    ui.launch(server_port=8888, server_name="localhost", show_error=True, share=False)
