<<<<<<< HEAD
"""A simple web interactive chat demo based on gradio."""

from argparse import ArgumentParser
from threading import Thread

import gradio as gr
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
=======
import gradio as gr
from argparse import ArgumentParser
from threading import Thread
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
>>>>>>> Yi-1.5/main
    TextIteratorStreamer,
)


<<<<<<< HEAD
class StopOnTokens(StoppingCriteria):
    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        stop_ids = (
            [2, 6, 7, 8],
        )  # "<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|im_sep|>"
        for stop_id in stop_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False


def parse_text(text):
    lines = text.split("\n")
    lines = [line for line in lines if line != ""]
    count = 0
    for i, line in enumerate(lines):
        if "```" in line:
            count += 1
            items = line.split("`")
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = f"<br></code></pre>"
        else:
            if i > 0:
                if count % 2 == 1:
                    line = line.replace("`", "\`")
                    line = line.replace("<", "&lt;")
                    line = line.replace(">", "&gt;")
                    line = line.replace(" ", "&nbsp;")
                    line = line.replace("*", "&ast;")
                    line = line.replace("_", "&lowbar;")
                    line = line.replace("-", "&#45;")
                    line = line.replace(".", "&#46;")
                    line = line.replace("!", "&#33;")
                    line = line.replace("(", "&#40;")
                    line = line.replace(")", "&#41;")
                    line = line.replace("$", "&#36;")
                lines[i] = "<br>" + line
    text = "".join(lines)
    return text


def predict(history, max_length, top_p, temperature):
    stop = StopOnTokens()
    messages = []
    for idx, (user_msg, model_msg) in enumerate(history):
        if idx == len(history) - 1 and not model_msg:
            messages.append({"role": "user", "content": user_msg})
            break
        if user_msg:
            messages.append({"role": "user", "content": user_msg})
        if model_msg:
            messages.append({"role": "assistant", "content": model_msg})

    print("\n\n====conversation====\n", messages)
    model_inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_tensors="pt"
    ).to(next(model.parameters()).device)
=======
def respond(
    message,
    history: list[tuple[str, str]],
    system_message,
    max_tokens,
    temperature,
    top_p,
):
    messages = [{"role": "system", "content": system_message}]

    for val in history:
        if val[0]:
            messages.append({"role": "user", "content": val[0]})
        if val[1]:
            messages.append({"role": "assistant", "content": val[1]})

    messages.append({"role": "user", "content": message})

    model_inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_tensors="pt"
    ).to(next(model.parameters()).device)

>>>>>>> Yi-1.5/main
    streamer = TextIteratorStreamer(
        tokenizer, timeout=60, skip_prompt=True, skip_special_tokens=True
    )
    generate_kwargs = {
        "input_ids": model_inputs,
        "streamer": streamer,
<<<<<<< HEAD
        "max_new_tokens": max_length,
        "do_sample": True,
        "top_p": top_p,
        "temperature": temperature,
        "stopping_criteria": StoppingCriteriaList([stop]),
        "repetition_penalty": 1.2,
    }
    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()

    for new_token in streamer:
        if new_token != "":
            history[-1][1] += new_token
            yield history


def main(args):
    with gr.Blocks() as demo:
        gr.Markdown(
            """\
<p align="center"><img src="https://raw.githubusercontent.com/01-ai/Yi/main/assets/img/Yi_logo_icon_light.svg" style="height: 80px"/><p>"""
        )
        gr.Markdown("""<center><font size=8>Yi-Chat Bot</center>""")
        gr.Markdown(
            """\
<center><font size=3>This WebUI is based on Yi-Chat, developed by 01-AI.</center>"""
        )
        gr.Markdown(
            """\
<center><font size=4>
Yi-34B-Chat <a style="text-decoration: none" href="https://huggingface.co/01-ai/Yi-34B-Chat">🤗</a> 
<a style="text-decoration: none" href="https://www.modelscope.cn/models/01ai/Yi-34B-Chat/summary">🤖</a>&nbsp  
&nbsp<a style="text-decoration: none" href="https://github.com/01-ai/Yi">Yi GitHub</a></center>"""
        )

        chatbot = gr.Chatbot()

        with gr.Row():
            with gr.Column(scale=4):
                with gr.Column(scale=12):
                    user_input = gr.Textbox(
                        show_label=False,
                        placeholder="Input...",
                        lines=10,
                        container=False,
                    )
                with gr.Column(min_width=32, scale=1):
                    submitBtn = gr.Button("🚀 Submit")
            with gr.Column(scale=1):
                emptyBtn = gr.Button("🧹 Clear History")
                max_length = gr.Slider(
                    0,
                    32768,
                    value=4096,
                    step=1.0,
                    label="Maximum length",
                    interactive=True,
                )
                top_p = gr.Slider(
                    0, 1, value=0.8, step=0.01, label="Top P", interactive=True
                )
                temperature = gr.Slider(
                    0.01, 1, value=0.6, step=0.01, label="Temperature", interactive=True
                )

        def user(query, history):
            return "", history + [[parse_text(query), ""]]

        submitBtn.click(
            user, [user_input, chatbot], [user_input, chatbot], queue=False
        ).then(predict, [chatbot, max_length, top_p, temperature], chatbot)
        user_input.submit(
            user, [user_input, chatbot], [user_input, chatbot], queue=False
        ).then(predict, [chatbot, max_length, top_p, temperature], chatbot)
        emptyBtn.click(lambda: None, None, chatbot, queue=False)

    demo.queue()

    demo.launch(
        server_name=args.server_name,
        server_port=args.server_port,
        inbrowser=args.inbrowser,
        share=args.share,
    )
=======
        "max_new_tokens": max_tokens,
        "do_sample": True,
        "top_p": top_p,
        "temperature": temperature,
        "repetition_penalty": 1.2,
        "eos_token_id": [tokenizer.eos_token_id],
    }
    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()
    response = ""
    for new_token in streamer:
        if new_token != "":
            response += new_token
            yield response


"""
For information on how to customize the ChatInterface, peruse the gradio docs: https://www.gradio.app/docs/chatinterface
"""
demo = gr.ChatInterface(
    respond,
    additional_inputs=[
        gr.Textbox(value="You are a friendly Chatbot.", label="System message"),
        gr.Slider(minimum=1, maximum=2048, value=512, step=1, label="Max new tokens"),
        gr.Slider(minimum=0.1, maximum=4.0, value=0.7, step=0.1, label="Temperature"),
        gr.Slider(
            minimum=0.1,
            maximum=1.0,
            value=0.95,
            step=0.05,
            label="Top-p (nucleus sampling)",
        ),
    ],
)
>>>>>>> Yi-1.5/main


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-c",
        "--checkpoint-path",
        type=str,
<<<<<<< HEAD
        default="01-ai/Yi-6B-Chat",
        help="Checkpoint name or path, default to %(default)r",
    )
    parser.add_argument(
        "--cpu-only", action="store_true", help="Run demo with CPU only"
    )
    parser.add_argument(
=======
        default="01-ai/Yi-1.5-34B-Chat",
        help="Checkpoint name or path, default to %(default)r",
    )
    parser.add_argument(
>>>>>>> Yi-1.5/main
        "--share",
        action="store_true",
        default=False,
        help="Create a publicly shareable link for the interface.",
    )
    parser.add_argument(
        "--inbrowser",
        action="store_true",
        default=True,
        help="Automatically launch the interface in a new tab on the default browser.",
    )
    parser.add_argument(
<<<<<<< HEAD
        "--server-port", type=int, default=8111, help="Demo server port."
=======
        "--server-port", type=int, default=8000, help="Demo server port."
>>>>>>> Yi-1.5/main
    )
    parser.add_argument(
        "--server-name", type=str, default="127.0.0.1", help="Demo server name."
    )

    args = parser.parse_args()

<<<<<<< HEAD
    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint_path, trust_remote_code=True
    )

    if args.cpu_only:
        device_map = "cpu"
    else:
        device_map = "auto"

    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint_path,
        device_map=device_map,
        torch_dtype="auto",
        trust_remote_code=True,
    ).eval()

    main(args)
=======
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_path, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint_path, device_map="auto", torch_dtype="auto"
    ).eval()

    demo.launch(
        server_name=args.server_name,
        server_port=args.server_port,
        inbrowser=args.inbrowser,
        share=args.share,
    )
>>>>>>> Yi-1.5/main
