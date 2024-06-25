# Prompt Lookup Decoding

**UPDATE 2**: This method is now available in [vLLM as well](https://docs.vllm.ai/en/stable/models/spec_decode.html#speculating-by-matching-n-grams-in-the-prompt) by setting `speculative_model="[ngram]"` 🥳

**UPDATE**: This has been [added to the transformers](https://twitter.com/joao_gante/status/1747322413006643259) library. Please see [this for a code example](https://pastebin.com/bms6XtR4), or simply add `prompt_lookup_num_tokens=10` to your `model.generate(...)` call.

Minimal implementation: See [demo notebook](./demo-pld.ipynb) or [colab](https://colab.research.google.com/drive/1ovjH1sg3lXWdm5Rx5EEukB9H_PFJVpJ4?usp=sharing)

**TLDR**: We modify speculative decoding where we replace the draft model with simple string matching in the prompt to generate candidate token sequences. This results in significant speedups (2x-4x) in input-grounded tasks, with no effect on output quality. This method can be used with any decoder model without model changes or external datastore, and with both greedy and sampling techniques.


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

- **Summarization**: CNN/Dailymail 100 examples
- **Context-QA**: 100 examples from [HAGRID](https://github.com/project-miracl/hagrid). We concatenate all the evidences to form the context and then do QA
- **Multi-turn chat**: [MT-bench](https://huggingface.co/datasets/HuggingFaceH4/mt_bench_prompts), all 80 examples. This isn't exactly input-grounded generation but gives an idea of performance on regular chat

## Results

### Summarization and Context-QA
On both summarization and context-QA, we get a relatively consistent 2.4x speedup (on average). The error bar is stddev, which shows there is quite a bit of variation depending on the example. Throughput of PLD was always more than that of greedy (or within margin of error) - I never saw it giving worse throughput than greedy on any example.

<img width="639" alt="image" src="https://github.com/apoorvumang/prompt-lookup-decoding/assets/1957903/df9aea26-3f49-473c-972a-0d9caa641b1e">

### Multi-turn chat
On MT-Bench, we see a similar gain on turn 1, but a much smaller gain on turn 0. This is expected - in the first turn, the algorithm can only match n-grams with its own output, since the prompt is pretty small. However this matching with self output gives measurable gains. Again, the error bars are stddev and I didn't see PLD giving worse throughput on any example.

<img width="535" alt="image" src="https://github.com/apoorvumang/prompt-lookup-decoding/assets/1957903/518420ac-b3d2-44af-8f76-b5656b2be6f4">

MT-Bench also has prompt categories. Some observations:
- roleplay has the worst gain. This is probably because there isn't many ngrams to copy, since each generation is sort of unique.
- coding has very high gain in 2nd turn, because there is lots of code copying
- in first turn, extraction has highest gain. This agrees with our hypothesis - in extraction there is definitely n-gram copying, and PLD should help

<img width="709" alt="image" src="https://github.com/apoorvumang/prompt-lookup-decoding/assets/1957903/5bd0b126-3e3f-453c-b04c-05e46b3619ce">

## TODOs/Thoughts/Future work
- There's probably better ways to do string matching than the current one, and there are several obvious things to improve eg. what to do when there are multiple matches? Whats the ideal length of continuation?
- We haven't yet tried sampling, although there's no reason it shouldn't work.
    - Here, one additional thing to test would be whether prompt lookup while sampling can affect hallucination rates, since this artifically increases probability of sampling exact sequences from input (this was suggest by my colleague Shwetha S)
- Testing actual FLOPs impact and tradeoffs is needed
- Also need to figure out best hyperparams - 3 and 10 were chosen on very little testing
- It would be an interesting challenge to design the "best lookup function" for decoding, could even be a competition?

## How to cite
```
@misc{saxena2023prompt,
    title = {Prompt Lookup Decoding},
    author = {Apoorv Saxena},
    year = {2023},
    month = {November},
    url = {https://github.com/apoorvumang/prompt-lookup-decoding/}
}
```
