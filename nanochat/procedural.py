"""Token tasks from Jiang et al. (2026), adapted to the packed NanoChat vocabulary.

The supervised suffix is expressed in *token* coordinates. Shift once when
packing, so the logit at the last delimiter predicts the first output token.
"""
from dataclasses import dataclass
import random



TRANSFORMS = ('identity', 'reverse', 'sort', 'set', 'delete', 'union', 'stack')
DYCK = ('dyck', 'dyck_shuffle')
TASKS = (*TRANSFORMS, *DYCK)


def _stack(rng, length, alphabet):
    stack, active, ops = [], set(), []
    symbols = set(range(1, len(alphabet) + 1))
    for i in range(length):
        if i == length - 1 and len(stack) > 1:
            push = False
        elif len(stack) <= 1:
            push = True
        elif i < length * 2 // 3:
            push = rng.random() < .75
        else:
            push = rng.random() >= .75
        if push and len(active) < len(symbols):
            available = list(symbols - active)
            symbol = available[rng.randint(len(available))]
            stack.append(symbol)
            active.add(symbol)
            ops.append(alphabet[symbol - 1])
        else:
            active.remove(stack.pop())
            ops.append(0)
    return ops, [alphabet[i - 1] for i in reversed(stack)]


def transform_example(task, length, alphabet, separator, rng):
    """Generate a complete example in native IDs and the first supervised token index."""
    if task == 'stack':
        prompt, output = _stack(rng, length, alphabet)
    else:
        prompt = [alphabet[i] for i in rng.randint(len(alphabet), size=length)]
        if task == 'identity':
            output = prompt
        elif task == 'reverse':
            output = prompt[::-1]
        elif task == 'sort':
            output = sorted(prompt)
        elif task == 'set':
            output = list(dict.fromkeys(prompt))
        elif task == 'union':
            half = length // 2
            prompt = prompt[:2 * half]
            output = list(dict.fromkeys(prompt))
            prompt = [*prompt[:half], separator, *prompt[half:]]
        elif task == 'delete':
            key = alphabet[rng.randint(len(alphabet))]
            output = prompt.copy()
            if key in output:  # The released generator removes the first match.
                output.remove(key)
            prompt = [*prompt, separator, key]
        else:
            raise ValueError(f'Unknown transformation {task}')
    tokens = [*prompt, separator, *output]
    if task == 'stack':
        tokens = tokens[:2 * length]  # The official stack input has exactly 2L slots.
    return tokens, len(prompt) + 1


def bracket_example(task, length, alphabet, rng):
    """64-Dyck or the bounded-depth shuffle with uniformly sampled active types."""
    k = len(alphabet) // 2
    stack, counts, tokens = [], [0] * k, []
    depth = 0
    for remaining in range(length, 0, -1):
        if task == 'dyck':
            opening = depth == 0 or (remaining > depth and rng.random() < .49 and depth < remaining - 1)
        else:
            opening = depth == 0 or (depth < 16 and rng.random() < .5)
        if opening:
            symbol = rng.randrange(k)
            if task == 'dyck':
                stack.append(symbol)
            counts[symbol] += 1
            depth += 1
            tokens.append(alphabet[symbol])
        else:
            symbol = stack.pop() if task == 'dyck' else rng.choice([i for i, n in enumerate(counts) if n])
            counts[symbol] -= 1
            depth -= 1
            tokens.append(alphabet[k + symbol])
    return tokens, 1


@dataclass(frozen=True)
class ProceduralData:
    task: str
    seed: int
    vocab_size: int
    context: int
    length: int

    def __post_init__(self):
        if self.task not in TASKS:
            raise ValueError(f'Unknown procedural task {self.task}')
        if self.vocab_size < 131 or not 0 <= self.seed < 2**32:
            raise ValueError('Procedural generation requires at least 131 native IDs and a uint32 seed')
        if self.length < 2:
            raise ValueError('Procedural input length must be at least two')
        if self.task in DYCK:
            if self.length % 2 or self.length > self.context:
                raise ValueError('Use an even bracket sequence length within the model context')
        else:
            if self.task == 'union' and self.length % 2:
                raise ValueError('Union uses two equal-length inputs')
            overhead = {'stack': 0, 'union': 2, 'delete': 3}.get(self.task, 1)
            if 2 * self.length + overhead > self.context:
                raise ValueError('A complete procedural example must fit in the model context')

    def batch(self, first_row, rows, split="train"):
        """Each row is a deterministic function of its index, independent of microbatch/resume."""
        import numpy as np

        if split not in ('train', 'dev'):
            raise ValueError(f'Unknown procedural split {split}')
        separator, pad = self.vocab_size - 2, self.vocab_size - 1
        pool = range(1, separator)
        inputs = np.full((rows, self.context), pad, dtype=np.int64)
        targets = np.full_like(inputs, pad)
        mask = np.zeros(inputs.shape, dtype=bool)
        # Bracket pairs must have the same meaning across examples.
        brackets = random.Random(self.seed).sample(pool, 128) if self.task in DYCK else None
        for b in range(rows):
            row = first_row + b
            namespace = 1 << 128 if split == 'dev' else 0
            mapping_rng = random.Random(namespace + (self.seed << 64) + row)
            rng = np.random.RandomState([self.seed, row, 1] if split == 'dev' else [self.seed, row])
            pos = 0
            while pos < self.context:
                if brackets is None:
                    # Preserve the reference collision rate while exposing all native symbol IDs.
                    alphabet = sorted(mapping_rng.sample(pool, 99))
                    tokens, first = transform_example(self.task, self.length, alphabet, separator, rng)
                else:
                    tokens, first = bracket_example(self.task, self.length, brackets, mapping_rng)
                end = pos + len(tokens)
                if end > self.context:
                    break
                inputs[b, pos:end] = tokens
                targets[b, pos:end - 1] = tokens[1:]
                mask[b, pos + first - 1:end - 1] = True
                pos = end + 1  # one native boundary token; no prediction across examples
        return inputs, targets, mask
