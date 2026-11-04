# Deep Learning from Scratch

The code for a blog series that teaches deep learning to C# and Python
developers: built by hand first, then with a framework, and every number
measured. Start with the [announcement](posts/00-announcing-the-series.md).

| # | Post | Code |
|---|---|---|
| 0 | [Announcing the series](posts/00-announcing-the-series.md) | |
| 1 | [A neuron is a weighted vote](posts/01-a-neuron-is-a-weighted-vote.md) | [`samples/01-neuron`](samples/01-neuron) (C#, Python) |
| 2 | [One neuron can't draw Cuba](posts/02-one-hidden-layer.md) | [`samples/02-hidden-layer`](samples/02-hidden-layer) (C#, Python) |
| 3 | [Blame flows backwards](posts/03-backprop.md) | [`samples/03-backprop`](samples/03-backprop) (C#, Python; PyTorch) |

## Running the samples

```bash
# C# (.NET 10)
cd samples/01-neuron/csharp && dotnet run -c Release

# Python 3.10+
pip install -r requirements.txt
cd samples/01-neuron/python && python main.py
```

`./check.sh` runs every sample in both languages and compares the output with
`samples/*/expected.txt`, the numbers quoted in the posts. CI runs it on every
push.

## License

Code: [MIT](LICENSE). Posts (`posts/`): [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Data: see [`data/README.md`](data/README.md).
