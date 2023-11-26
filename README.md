# Prompt Lookup Decoding

Minimal implementation: See [demo notebook](./demo-pld.ipynb) or [colab](https://colab.research.google.com/drive/1ovjH1sg3lXWdm5Rx5EEukB9H_PFJVpJ4?usp=sharing)

**TLDR**: We modify speculative decoding where we replace the draft model with a simple heuristic based on n-gram overlap between LLM input and output. This results in significant speedups (2x-4x) in input-grounded tasks, with no effect on output quality. This method can be used with any decoder model, and with both greedy and sampling techniques.


https://github.com/apoorvumang/prompt-lookup-decoding/assets/1957903/e908de89-ce5c-4156-8ef1-21f169dc1c8f

Coloured token indicate that multiple tokens were generated in a single step.


## Method

**Intuition**: In several LLM use cases where you're doing _input grounded generation_ (summarization, document QA, multi-turn chat, code editing), there is high n-gram overlap between LLM input (prompt) and LLM output. This could be entity names, phrases, or code chunks that the LLM directly copies from the input while generating the output. Prompt lookup exploits this pattern to speed up autoregressive decoding in LLMs.

Here's an animation explaining with an example (for information on how speculative decoding itself works/gives speedup, please see this excellent [huggingface blog](https://huggingface.co/blog/assisted-generation)):

https://github.com/apoorvumang/prompt-lookup-decoding/assets/1957903/10c3728b-2d50-4205-a758-478e51425793

This is the "prompt lookup" function:

```
def find_candidate_pred_tokens(input_ids, max_ngram_size=3, num_pred_tokens=10):
    input_length = input_ids.size(1)

    for ngram_size in range(max_ngram_size, 0, -1):
        # Extract the last n tokens as our search ngram
        ngram = input_ids[0, -ngram_size:].tolist()

        # Create sliding windows of size ngram_size
        windows = input_ids.unfold(dimension=1, size=ngram_size, step=1)

        # Convert ngram to a tensor for comparison
        ngram_tensor = torch.tensor(ngram, device=input_ids.device).unsqueeze(0)

        # Find where the windows match the ngram
        matches = (windows == ngram_tensor).all(dim=2)

        # Get the indices of matches
        match_indices = matches.nonzero(as_tuple=True)[1]

        # Iterate through match indices to find a valid continuation
        for idx in match_indices:
            start_idx = idx + ngram_size
            end_idx = start_idx + num_pred_tokens
            # Ensure we don't go beyond the length of input_ids and avoid self-match
            if end_idx <= input_length and start_idx < input_length - ngram_size:
                return input_ids[0, start_idx:end_idx]

    # If no match is found, return an empty tensor
    return torch.tensor([], dtype=torch.long, device=input_ids.device)

```


Implementation-wise, we modify [speculative decoding](https://arxiv.org/pdf/2302.01318.pdf) (aka assisted generation in hf transformers) by swapping out the “draft model” with this function.

Input to this function is the same as to the draft model - all the tokens till the current generation step (`input_ids`). It then tries to match last few tokens to somewhere earlier in the prompt. If found, it returns the next-k token continuation as `candidate_input_ids` or candidate sequence. The 2 parameters are `max_ngram_size`, which is the maximum ngram to use when looking for matches in the prompt. `num_pred_tokens` is the candidate sequence length to return after match is found.


## Experimental setup

- **GPU**: Single A100 40GB
- **Model**: Mistral-7B-Instruct-v0.1
- **Decoding Type**: Greedy decoding
- **Hyperparams**: Matching max n-gram size = 3, Continuation length = 10

## Datasets

We experiment on 3 datasets, and compare with simple greedy decoding as a baseline. We focus on "input-grounded" tasks where we expect high overlap between input and output - summarization, context-based QA and multi-turn chat.
