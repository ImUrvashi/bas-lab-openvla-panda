"""
llm_frontend.py -- Llama 3 front end: free-text operator request -> one of the five station tasks.

OpenVLA only knows the five instruction sentences it was trained on. The front end reads whatever the operator
types, picks the task (or none) and writes a short reply; the task's trained sentence then goes to OpenVLA.

    fe = LlamaFrontEnd("meta-llama/Meta-Llama-3-8B-Instruct")      # 4-bit, about 6 GB of GPU memory
    fe.interpret("the coolant line is leaking, shut it off")
        -> {"task": 3, "instruction": "Close the coolant valve.", "reply": "..."}
"""
import json
import re

TASKS = {
    1: "Pick up the red cube and place it on the yellow marker.",
    2: "Put the tool in the drawer and close it.",
    3: "Close the coolant valve.",
    4: "Insert the filter cartridge into the air scrubber.",
    5: "Put the sample vial in the container and close the lid.",
}

SYSTEM = """You are the voice interface of a Panda robot arm on a space-station lab bench.
The robot can do exactly five tasks:
1. Pick up the red sample cube and place it on the yellow processing marker.
2. Put the green tool into the cabinet drawer and close the drawer.
3. Close the coolant valve (turn its red lever a quarter turn).
4. Insert the teal filter cartridge into the air scrubber.
5. Put the pink sample vial into the purple container and close the lid.
Read the operator's request and answer with JSON only, no other text:
{"task": <1-5, or 0 if no task fits>, "reply": "<one short sentence to the operator>"}
Use 0 when the request asks for something else, several tasks at once, or is unclear; the reply then says
briefly what you can do. Never invent a sixth task."""

EXAMPLES = [
    ("could you stash the wrench away in the drawer", {"task": 2, "reply": "Stowing the tool in the drawer and closing it."}),
    ("make me a coffee", {"task": 0, "reply": "I can't make coffee; I can move the cube, stow the tool, close the valve, "
                                             "insert the filter or seal the vial."}),
    ("the air quality is dropping, replace the scrubber filter", {"task": 4, "reply": "Inserting the filter cartridge into the scrubber."}),
]


def parse_reply(text):
    """Model text -> {"task", "instruction", "reply"}; task 0 if the output is not usable."""
    m = re.search(r"\{.*\}", text, re.S)
    try:
        obj = json.loads(m.group(0)) if m else {}
        task = int(obj.get("task", 0))
    except (ValueError, TypeError):
        obj, task = {}, 0
    task = task if task in TASKS else 0
    reply = str(obj.get("reply", "")).strip() or ("OK." if task else "Sorry, I did not understand that.")
    return {"task": task, "instruction": TASKS.get(task, ""), "reply": reply[:200]}


class LlamaFrontEnd:
    def __init__(self, model_id, device="cuda:0"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, device_map={"": device}, torch_dtype=torch.bfloat16,
            quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16))
        self.model.eval()
        self.device = device
        self.model_id = model_id

    def interpret(self, text):
        msgs = [{"role": "system", "content": SYSTEM}]
        for q, a in EXAMPLES:
            msgs += [{"role": "user", "content": q}, {"role": "assistant", "content": json.dumps(a)}]
        msgs.append({"role": "user", "content": text[:500]})
        ids = self.tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(self.device)
        stop = [self.tok.eos_token_id, self.tok.convert_tokens_to_ids("<|eot_id|>")]
        with self.torch.no_grad():
            out = self.model.generate(ids, max_new_tokens=80, do_sample=False, eos_token_id=stop,
                                      pad_token_id=self.tok.eos_token_id)
        return parse_reply(self.tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True))
