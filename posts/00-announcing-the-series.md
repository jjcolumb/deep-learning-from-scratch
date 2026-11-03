# Deep Learning from Scratch: a series for C# and Python developers

I grew up in Moa, a small mining town in Holguín, on the eastern side of Cuba,
where the earth is red from the nickel. In Cuba you learn early that if
something is broken, nobody is coming to fix it for you. You open it up, you
figure out how it works, and you make it run again with whatever you have.
Every Cuban knows somebody who has kept a 1950s car alive for seventy years
that way.

I've been using AI every day, like everyone. It writes code with me, answers my
questions, and summarizes my meetings. And at some point it started to bother
me that I had no idea what was happening inside. For a developer, that's an
uncomfortable feeling. We're supposed to understand our tools.

So I did what Cubans do with a car that won't start. I opened the hood.

## What I found under the hood

For the last months I've been learning deep learning properly. I built every
piece by hand first, and only then let a framework do it for me. I measured
everything instead of trusting explanations, including the explanations in
the material I was learning from. More than once, the measurement disagreed
with the explanation, and those were the moments I learned the most.

What surprised me most is how *small* the core idea is. A neuron is a weighted
vote: a few multiplications and an addition. Everything else, from the network
that recognizes your face to the one that writes this paragraph back to you in
another language, is that same small idea stacked and wired together cleverly.
You don't need a PhD to understand it. You need someone to walk you through it
without skipping the parts that matter.

That's what this series is for.

## Who it's for

It's for developers who know C# or Python and have **no machine-learning
background**. If you're comfortable with a loop and an array, you have enough.
Math shows up only where it earns its place, and every term gets explained in
plain words the first time it appears.

## How we'll do it

Every post starts with a story, the "why", before the "how". Then we build the
idea in code small enough to run on a laptop in a minute. Then I'll show you
what went wrong when I tried it, because that's usually where the real lesson
is.

Three promises:

- **C# first, then Python.** For everything we build from scratch (neurons,
  networks, backpropagation, optimizers, convolutions, recurrent networks),
  you'll see the C# version first and the Python version after. Both print the
  same numbers, digit for digit, and an automated check makes sure they keep
  doing so.
- **Every number is measured.** If a post says 96.3%, there's code in the repo
  that prints 96.3%. Nothing is copied from somewhere else, and nothing is
  typed from memory.
- **No magic boxes for C# developers.** When we move to a framework for the
  bigger models, the code will be Python, because that's where the ecosystem
  lives today. But each of those posts will have a "For C# developers"
  section with the equivalent steps in [TorchSharp](https://github.com/dotnet/TorchSharp),
  .NET's version of PyTorch.

## Where we're going

We'll follow two threads all the way through. The first is **photos of
frogs**: one small image dataset that we'll take from a single neuron to
a network trained on millions of photos, watching the accuracy climb one idea
at a time. The second is **dates**: teaching a model to turn
"saturday august 20 1977" into `1977-08-20`, first with a recurrent network
and then with a Transformer, the architecture behind today's chatbots. That
way we can compare the two directly on the same problem.

Here's the map. It'll grow and change as we go.

**Part 1: How a network learns**
1. A neuron is a weighted vote *(published)*
2. One hidden layer: when a straight line isn't enough
3. Backpropagation by hand, checking it, then letting the framework do it

**Part 2: Making training work**

4. Initialization: why starting everyone equal fails (the libreta problem)
5. Overfitting: memorizing the exam vs. understanding the course
6. Optimizers: mini-batches, momentum, Adam and learning-rate decay

**Part 3: Vision**

7. Convolutions from scratch
8. Small and normalized beats big: batch normalization
9. ResNets: why skip connections let you go deep
10. Transfer learning: borrowing a network trained on millions of photos
11. Object detection: boxes, overlap and removing duplicates
12. Segmentation: a label for every pixel
13. Face recognition: one photo per person is enough
14. Style transfer: training the image instead of the network

**Part 4: Sequences and language**

15. Recurrent networks and LSTMs from scratch
16. Models that write, and when "creative" really means "copied"
17. Word vectors: meaning as geometry, and its biases
18. Classifying sentences: when averaging beats an LSTM
19. Attention: translating dates
20. Transformers: the engine inside today's AI

## This is a series in progress

I'm writing these as I go. New posts and their code will show up here and in
the repo as each one is ready, so this isn't a finished book. It's more like
an open workshop, and you're welcome to pull up a chair.

All the code lives at
**[github.com/jjcolumb/deep-learning-from-scratch](https://github.com/jjcolumb/deep-learning-from-scratch)**.
Star or watch it to see new samples as they land, and keep an eye on this
blog for the next post. If something isn't clear, or you think I got
something wrong, tell me. Finding out where an explanation and a measurement
disagree is the whole spirit of this series.

The first post is already up: [**A neuron is a weighted vote**](01-a-neuron-is-a-weighted-vote.md),
where a corner argument about Santiago vs Industriales turns into the basic
building block of every AI model you've ever used.

Let's open the hood.
