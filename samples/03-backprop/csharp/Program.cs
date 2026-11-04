// Post 3 -- backpropagation for any number of layers, and how to check it.
//
// Tiny photos, frog or not. We write the backward pass once, for any depth,
// train networks of different depths, then check the gradients by nudging each
// weight and watching the cost move. A bug that crashes nothing gets caught.
//
//     python ../../../data/make_frogs.py    # once: builds the photo files
//     dotnet run -c Release

// walk up from the build folder until we find the repo's data/ folder
var dir = new DirectoryInfo(AppContext.BaseDirectory);
while (!File.Exists(Path.Combine(dir.FullName, "data", "make_frogs.py")))
    dir = dir.Parent ?? throw new FileNotFoundException("data/ folder not found");
string data = Path.Combine(dir.FullName, "data");
if (!File.Exists(Path.Combine(data, "frogs_train.bin")))
    throw new FileNotFoundException("run data/make_frogs.py first");

// --- load: 3,072 pixel values per photo -> frog (1) or not (0) -----------
var (xTrain, yTrain) = Load("frogs_train.bin");
var (xTest, yTest) = Load("frogs_test.bin");

// centre every pixel: subtract its average over the training photos
var mean = new double[3072];
foreach (var x in xTrain)
    for (int k = 0; k < 3072; k++)
        mean[k] += x[k];
for (int k = 0; k < 3072; k++)
    mean[k] /= xTrain.Length;
foreach (var x in xTrain.Concat(xTest))
    for (int k = 0; k < 3072; k++)
        x[k] -= mean[k];

Console.WriteLine($"{yTrain.Length} training photos, {yTest.Length} test photos, half of each are frogs\n");
Console.WriteLine($"{"model",-34}{"train",8}{"test",8}");
foreach (var (label, sizes, bug) in new (string, int[], bool)[] {
             ("one neuron", [3072, 1], false),
             ("1 hidden layer (20)", [3072, 20, 1], false),
             ("3 hidden layers (20, 7, 5)", [3072, 20, 7, 5, 1], false),
             ("3 hidden layers, backprop bug", [3072, 20, 7, 5, 1], true) })
{
    var net = new Network(sizes);
    net.Train(xTrain, yTrain, bug: bug);
    Console.WriteLine($"{label,-34}{net.Accuracy(xTrain, yTrain),7:F1}%{net.Accuracy(xTest, yTest),7:F1}%");
}

int[] deep = [3072, 20, 7, 5, 1];
var xFew = xTrain[..16];
var yFew = yTrain[..16];
foreach (var (title, bug) in new[] { ("correct backprop", false), ("backprop bug", true) })
{
    var (perLayer, overall, count) = GradientCheck.Run(deep, xFew, yFew, bug: bug);
    Console.WriteLine($"\ngradient check, {title} ({count} weights, 16 photos): difference {overall:0.0e+00}");
    for (int l = 0; l < perLayer.Length; l++)
        Console.WriteLine($"  layer {l + 1}: {perLayer[l]:0.0e+00}{(perLayer[l] > 1e-5 ? "   <- wrong" : "")}");
}

Console.WriteLine("\nsize of the nudge vs. difference (correct backprop)");
foreach (double epsilon in new[] { 1e-1, 1e-3, 1e-5, 1e-7, 1e-9, 1e-11, 1e-13 })
    Console.WriteLine($"  {epsilon:0e+00}: {GradientCheck.Run(deep, xFew, yFew, epsilon).overall:0.0e+00}");

(double[][], double[]) Load(string name)
{
    var raw = File.ReadAllBytes(Path.Combine(data, name));
    int n = raw.Length / (1 + 3072);
    var x = new double[n][];
    var y = new double[n];
    for (int i = 0; i < n; i++)
    {
        int at = i * (1 + 3072);
        y[i] = raw[at];
        x[i] = new double[3072];
        for (int k = 0; k < 3072; k++)
            x[i][k] = raw[at + 1 + k] / 255.0;
    }
    return (x, y);
}

/// <summary>sizes = [3072, 20, 7, 5, 1]: inputs, hidden layers, one output.</summary>
class Network
{
    public readonly int[] Sizes;
    public readonly double[][] W;   // layer l: Sizes[l] x Sizes[l+1], row by row
    public readonly double[][] B;

    public Network(int[] sizes, ulong seed = 42)
    {
        Sizes = sizes;
        var rng = new SplitMix64(seed);
        W = new double[sizes.Length - 1][];
        B = new double[sizes.Length - 1][];
        for (int l = 0; l < W.Length; l++)
        {
            int n = sizes[l], h = sizes[l + 1];
            double scale = Math.Sqrt(6.0 / n);   // keeps signals from growing layer by layer (part 4)
            W[l] = new double[n * h];
            for (int i = 0; i < n * h; i++)
                W[l][i] = rng.Uniform() * scale;
            B[l] = new double[h];
        }
    }

    /// <summary>Every layer's output, from the photos (A[0]) to the answers (A[^1]).</summary>
    public List<double[][]> Forward(double[][] X)
    {
        var A = new List<double[][]> { X };
        for (int l = 0; l < W.Length; l++)
        {
            int n = Sizes[l], h = Sizes[l + 1];
            bool last = l == W.Length - 1;
            var output = new double[X.Length][];
            for (int i = 0; i < X.Length; i++)
            {
                var a = A[l][i];
                var z = (double[])B[l].Clone();
                for (int k = 0; k < n; k++)
                    for (int j = 0; j < h; j++)
                        z[j] += a[k] * W[l][k * h + j];
                for (int j = 0; j < h; j++)
                    z[j] = last ? 1 / (1 + Math.Exp(-z[j])) : (z[j] > 0 ? z[j] : 0.0);   // sigmoid or ReLU
                output[i] = z;
            }
            A.Add(output);
        }
        return A;
    }

    public double Cost(double[][] X, double[] y)
    {
        var p = Forward(X)[^1];
        double sum = 0;
        for (int i = 0; i < y.Length; i++)
            sum += y[i] * Math.Log(p[i][0]) + (1 - y[i]) * Math.Log(1 - p[i][0]);
        return -sum / y.Length;
    }

    /// <summary>The gradient of the cost for every weight, last layer first.</summary>
    public (double[] dW, double[] dB)[] Backward(List<double[][]> A, double[] y, bool bug = false)
    {
        int m = y.Length;
        var dZ = A[^1].Select((p, i) => new[] { p[0] - y[i] }).ToArray();   // how wrong each answer was
        var grads = new (double[], double[])[W.Length];
        for (int l = W.Length - 1; l >= 0; l--)
        {
            int n = Sizes[l], h = Sizes[l + 1];
            var dW = new double[n * h];
            var dB = new double[h];
            for (int i = 0; i < m; i++)
                for (int k = 0; k < n; k++)
                    for (int j = 0; j < h; j++)
                        dW[k * h + j] += A[l][i][k] * dZ[i][j];      // blame for each weight
            for (int i = 0; i < m; i++)
                for (int j = 0; j < h; j++)
                    dB[j] += dZ[i][j];
            for (int x = 0; x < n * h; x++) dW[x] /= m;
            for (int j = 0; j < h; j++) dB[j] /= m;
            grads[l] = (dW, dB);
            if (l == 0)
                break;
            var below = new double[m][];
            for (int i = 0; i < m; i++)
            {
                below[i] = new double[n];
                for (int k = 0; k < n; k++)
                {
                    // pass the blame one layer down: through the weights...
                    double dA = 0;
                    for (int j = 0; j < h; j++)
                        dA += dZ[i][j] * W[l][k * h + j];
                    // ...and through ReLU's slope: 1 where the neuron fired, 0 where it didn't
                    below[i][k] = bug ? dA : dA * (A[l][i][k] > 0 ? 1.0 : 0.0);
                }
            }
            dZ = below;
        }
        return grads;
    }

    public void Train(double[][] X, double[] y, int epochs = 40, int batch = 64, double learningRate = 0.003, bool bug = false)
    {
        var rng = new SplitMix64(7);
        var order = Enumerable.Range(0, y.Length).ToArray();
        for (int epoch = 0; epoch < epochs; epoch++)
        {
            for (int i = order.Length - 1; i >= 1; i--)               // shuffle (Fisher-Yates)
            {
                int j = rng.Below(i + 1);
                (order[i], order[j]) = (order[j], order[i]);
            }
            for (int start = 0; start < order.Length; start += batch)
            {
                var rows = order[start..Math.Min(start + batch, order.Length)];
                var grads = Backward(Forward(rows.Select(r => X[r]).ToArray()), rows.Select(r => y[r]).ToArray(), bug);
                for (int l = 0; l < W.Length; l++)
                {
                    for (int x = 0; x < W[l].Length; x++) W[l][x] -= learningRate * grads[l].dW[x];
                    for (int j = 0; j < B[l].Length; j++) B[l][j] -= learningRate * grads[l].dB[j];
                }
            }
        }
    }

    public double Accuracy(double[][] X, double[] y)
    {
        var p = Forward(X)[^1];
        int right = Enumerable.Range(0, y.Length).Count(i => p[i][0] > 0.5 == (y[i] == 1));
        return 100 * ((double)right / y.Length);
    }
}

static class GradientCheck
{
    /// <summary>Backprop's gradients vs. ones measured by nudging each weight.
    /// Returns the relative difference for each layer and for all of them.</summary>
    public static (double[] perLayer, double overall, int count) Run(
        int[] sizes, double[][] X, double[] y, double epsilon = 1e-7, bool bug = false, int weightsInFirstLayer = 100)
    {
        var net = new Network(sizes);
        var grads = net.Backward(net.Forward(X), y, bug);
        var rng = new SplitMix64(3);
        var claimed = new List<double>();
        var measured = new List<double>();
        var layerOf = new List<int>();
        for (int l = 0; l < net.W.Length; l++)
        {
            int n = sizes[l], h = sizes[l + 1];
            var cells = new List<(double[] target, double[] grad, int index)>();
            if (l == 0)
                for (int c = 0; c < weightsInFirstLayer; c++)
                {
                    int k = rng.Below(n), j = rng.Below(h);
                    cells.Add((net.W[l], grads[l].dW, k * h + j));
                }
            else
                for (int x = 0; x < n * h; x++)
                    cells.Add((net.W[l], grads[l].dW, x));
            for (int j = 0; j < h; j++)
                cells.Add((net.B[l], grads[l].dB, j));

            foreach (var (target, grad, index) in cells)
            {
                double keep = target[index];
                target[index] = keep + epsilon;
                double plus = net.Cost(X, y);
                target[index] = keep - epsilon;
                double minus = net.Cost(X, y);
                target[index] = keep;
                claimed.Add(grad[index]);
                measured.Add((plus - minus) / (2 * epsilon));
                layerOf.Add(l);
            }
        }

        double Difference(Func<int, bool> keep)
        {
            double diff = 0, g = 0, a = 0;
            for (int i = 0; i < claimed.Count; i++)
                if (keep(layerOf[i]))
                {
                    diff += (claimed[i] - measured[i]) * (claimed[i] - measured[i]);
                    g += claimed[i] * claimed[i];
                    a += measured[i] * measured[i];
                }
            return Math.Sqrt(diff) / (Math.Sqrt(g) + Math.Sqrt(a));
        }

        var perLayer = Enumerable.Range(0, sizes.Length - 1).Select(l => Difference(x => x == l)).ToArray();
        return (perLayer, Difference(_ => true), claimed.Count);
    }
}

/// <summary>A tiny random generator, written by hand so the Python version
/// makes exactly the same choices.</summary>
class SplitMix64(ulong state)
{
    ulong Next()
    {
        state += 0x9E3779B97F4A7C15;
        ulong z = state;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EB;
        return z ^ (z >> 31);
    }

    public double Uniform() => (Next() >> 11) / (double)(1UL << 53) * 2 - 1;   // between -1 and 1

    public int Below(int n) => (int)(Next() % (ulong)n);                      // 0 to n - 1
}
