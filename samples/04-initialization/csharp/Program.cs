// Post 4 -- where the weights start decides whether a network learns at all.
//
// The frog network from part 3, trained exactly the same way six times. The only
// thing that changes is the starting weights. Then a 10-layer network, where
// the difference grows.
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

// --- the six ways to start ------------------------------------------------
// each one: a layer with n inputs -> how to pick one starting weight
var starts = new Dictionary<string, Func<SplitMix64, int, double>>
{
    ["all zeros"] = (rng, n) => 0.0,
    ["all equal (0.01)"] = (rng, n) => 0.01,                      // the libreta: same ration for everyone
    ["random, tiny (0.01)"] = (rng, n) => rng.Uniform() * 0.01,
    ["random, big (1)"] = (rng, n) => rng.Uniform(),
    ["random, 1/sqrt(n)"] = (rng, n) => rng.Uniform() * (1 / Math.Sqrt(n)),   // PyTorch's default
    ["random, sqrt(6/n)"] = (rng, n) => rng.Uniform() * Math.Sqrt(6.0 / n),   // ours, since part 3
};

Console.WriteLine($"{yTrain.Length} training photos, {yTest.Length} test photos, half of each are frogs");
foreach (var (title, sizes, names) in new (string, int[], string[])[] {
             ("3 hidden layers (20, 7, 5)", [3072, 20, 7, 5, 1], starts.Keys.ToArray()),
             ("10 hidden layers of 20", [3072, .. Enumerable.Repeat(20, 10), 1],
              ["random, tiny (0.01)", "random, 1/sqrt(n)", "random, sqrt(6/n)"]) })
{
    Console.WriteLine($"\n{title}");
    Console.WriteLine($"{"starting weights",-24}{"train",8}{"test",8}   signal at the start, layer by layer");
    foreach (var name in names)
    {
        var net = new Network(sizes, starts[name]);
        var signal = net.Signal(xTrain[..500]);
        net.Train(xTrain, yTrain);
        Console.WriteLine($"{name,-24}{net.Accuracy(xTrain, yTrain),7:F1}%{net.Accuracy(xTest, yTest),7:F1}%   " +
                          string.Join(" ", signal.Select(s => s.ToString("0e+00"))));
    }
}

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

    public Network(int[] sizes, Func<SplitMix64, int, double> start, ulong seed = 42)
    {
        Sizes = sizes;
        var rng = new SplitMix64(seed);
        W = new double[sizes.Length - 1][];
        B = new double[sizes.Length - 1][];
        for (int l = 0; l < W.Length; l++)
        {
            int n = sizes[l], h = sizes[l + 1];
            W[l] = new double[n * h];
            for (int i = 0; i < n * h; i++)
                W[l][i] = start(rng, n);
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

    /// <summary>The gradient of the cost for every weight, last layer first.</summary>
    public (double[] dW, double[] dB)[] Backward(List<double[][]> A, double[] y)
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
                    double dA = 0;
                    for (int j = 0; j < h; j++)
                        dA += dZ[i][j] * W[l][k * h + j];
                    below[i][k] = dA * (A[l][i][k] > 0 ? 1.0 : 0.0);
                }
            }
            dZ = below;
        }
        return grads;
    }

    public void Train(double[][] X, double[] y, int epochs = 40, int batch = 64, double learningRate = 0.003)
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
                var grads = Backward(Forward(rows.Select(r => X[r]).ToArray()), rows.Select(r => y[r]).ToArray());
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

    /// <summary>How big each hidden layer's outputs are (root mean square).</summary>
    public double[] Signal(double[][] X)
    {
        var A = Forward(X);
        var sizes = new double[A.Count - 2];
        for (int l = 1; l < A.Count - 1; l++)
        {
            double squares = 0;
            int count = 0;
            foreach (var row in A[l])            // photo by photo, neuron by neuron
                foreach (var a in row)
                {
                    squares += a * a;
                    count++;
                }
            sizes[l - 1] = Math.Sqrt(squares / count);
        }
        return sizes;
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
