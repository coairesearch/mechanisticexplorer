# %%
import torch
import numpy as np
import matplotlib.pyplot as plt
from nnsight import LanguageModel
import transformers

import nnsight
from nnsight import CONFIG

CONFIG.API.HOST = "localhost:5001"
CONFIG.API.SSL = False
CONFIG.API.APIKEY = "0Bb6oUQxj2TuPtlrTkny"
model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
model_name2 = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"

# At the beginning of your script, after imports
torch.set_grad_enabled(False) 
# %%
## Simple trace Example for the firs generated token for the last layer.
model = LanguageModel(model_name, device_map="auto")
# %%
prompt= "The Eiffel Tower is in the city of"
layers = model.model.layers
probs_layers = []

with model.trace(remote=True) as tracer:
    with tracer.invoke(prompt) as invoker:
        for layer_idx, layer in enumerate(layers):
            # Process layer output through the model's head and layer normalization
            layer_output = model.lm_head(model.model.norm(layer.output[0]))

            # Apply softmax to obtain probabilities and save the result
            probs = torch.nn.functional.softmax(layer_output, dim=-1).save()
            probs_layers.append(probs)

probs = torch.cat([probs.value for probs in probs_layers])

# Find the maximum probability and corresponding tokens for each position
max_probs, tokens = probs.max(dim=-1)

# Decode token IDs to words for each layer
words = [[model.tokenizer.decode(t.cpu()).encode("unicode_escape").decode() for t in layer_tokens]
    for layer_tokens in tokens]

# Access the 'input_ids' attribute of the invoker object to get the input words
input_words = [model.tokenizer.decode(t) for t in invoker.inputs[0][0]["input_ids"][0]]
# %%
# %%
import plotly.express as px
import plotly.io as pio

#pio.renderers.default = "plotly_mimetype+notebook_connected+notebook"

fig = px.imshow(
    max_probs.detach().cpu().to(torch.float32).numpy(),
    x=input_words,
    y=list(range(len(words))),
    color_continuous_scale=px.colors.diverging.RdYlBu_r,
    color_continuous_midpoint=0.50,
    text_auto=True,
    labels=dict(x="Input Tokens", y="Layers", color="Probability")
)

fig.update_layout(
    title='Logit Lens Visualization',
    xaxis_tickangle=0
)

fig.update_traces(text=words, texttemplate="%{text}")
fig.show()
# %%
prompt = 'The Eiffel Tower is in the city of'
n_new_tokens = 10
with model.generate(prompt, max_new_tokens=n_new_tokens, remote=True) as tracer:
    out = model.generator.output.save()

decoded_prompt = model.tokenizer.decode(out[0][0:-n_new_tokens].cpu())
decoded_answer = model.tokenizer.decode(out[0][-n_new_tokens:].cpu())

print("Prompt: ", decoded_prompt)
print("Generated Answer: ", decoded_answer)
# %%
