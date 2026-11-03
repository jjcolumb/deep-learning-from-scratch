// Post 2 -- one neuron can't draw Cuba. A hidden layer can.
//
// Points around Cuba, half on land and half at sea. We train a single neuron,
// a hidden layer without a bend, and hidden layers of different sizes with one.
//
//     dotnet run -c Release

using System.Globalization;

// walk up from the build folder until we find the repo's data/ folder
var dir = new DirectoryInfo(AppContext.BaseDirectory);
while (!File.Exists(Path.Combine(dir.FullName, "data", "cuba.csv")))
    dir = dir.Parent ?? throw new FileNotFoundException("data/cuba.csv not found");

// --- load: longitude, latitude -> land (1) or sea (0) --------------------
var rows = File.ReadLines(Path.Combine(dir.FullName, "data", "cuba.csv")).Skip(1)
    .Select(line => line.Split(',').Select(s => double.Parse(s, CultureInfo.InvariantCulture)).ToArray())
    .ToArray();

var train = rows.Where((_, i) => i % 4 != 3).ToArray();    // every 4th point is the exam
var test = rows.Where((_, i) => i % 4 == 3).ToArray();
var mean = new double[2];
var std = new double[2];
for (int j = 0; j < 2; j++)
{
    double sum = 0, squares = 0;
    foreach (var r in train) sum += r[j];
    mean[j] = sum / train.Length;
    foreach (var r in train) squares += (r[j] - mean[j]) * (r[j] - mean[j]);
    std[j] = Math.Sqrt(squares / train.Length);
}
var (xTrain, yTrain) = Split(train);
var (xTest, yTest) = Split(test);

Console.WriteLine($"{train.Length} training points, {test.Length} test points, " +
                  $"{100 * yTest.Average():F1}% of the test points are land\n");
Console.WriteLine($"{"model",-28}{"train",8}{"test",8}");
var models = new List<(string, int, bool)> { ("one neuron", 0, true), ("10 hidden, no bend", 10, false) };
foreach (int h in new[] { 1, 2, 3, 4, 6, 10, 20, 50 })
    models.Add(($"{h} hidden, tanh", h, true));
foreach (var (label, hidden, bend) in models)
{
    var net = new Network(hidden, bend);
    net.Train(xTrain, yTrain, steps: 20_000, learningRate: 1.0);
    Console.WriteLine($"{label,-28}{net.Accuracy(xTrain, yTrain),7:F1}%{net.Accuracy(xTest, yTest),7:F1}%");
}

(double[][], double[]) Split(double[][] r) =>
    (r.Select(x => new[] { (x[0] - mean[0]) / std[0], (x[1] - mean[1]) / std[1] }).ToArray(),
     r.Select(x => x[2]).ToArray());

/// <summary>hidden = 0 is a single neuron (post 1). bend = false removes tanh.</summary>
class Network
{
    readonly int hidden;
    readonly bool bend;
    readonly double[,] W1;      // 2 inputs x hidden
    readonly double[] b1;
    readonly double[] W2;       // hidden (or 2 inputs, for a single neuron) -> 1 output
    double b2;

    public Network(int hidden, bool bend)
    {
        this.hidden = hidden;
        this.bend = bend;
        var rng = new SplitMix64(42);
        W1 = new double[2, hidden];
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < hidden; j++)
                W1[i, j] = rng.Uniform();
        b1 = new double[hidden];
        W2 = new double[hidden == 0 ? 2 : hidden];
        if (hidden > 0)
            for (int j = 0; j < hidden; j++)
                W2[j] = rng.Uniform();
    }

    // returns the hidden activations (or the inputs, for a single neuron) and the output
    (double[] a1, double p) Forward(double[] x)
    {
        if (hidden == 0)
            return (x, Sigmoid(W2[0] * x[0] + W2[1] * x[1] + b2));
        var a1 = new double[hidden];
        double z2 = b2;
        for (int j = 0; j < hidden; j++)
        {
            double z1 = x[0] * W1[0, j] + x[1] * W1[1, j] + b1[j];   // each hidden neuron's vote
            a1[j] = bend ? Tanh(z1) : z1;                              // ...bent by tanh, or not
            z2 += a1[j] * W2[j];                                       // the output neuron votes on their votes
        }
        return (a1, Sigmoid(z2));
    }

    public void Train(double[][] X, double[] y, int steps, double learningRate)
    {
        int n = y.Length;
        for (int step = 0; step < steps; step++)
        {
            var dW1 = new double[2, hidden];
            var db1 = new double[hidden];
            var dW2 = new double[W2.Length];
            double db2 = 0;
            for (int i = 0; i < n; i++)
            {
                var (a1, p) = Forward(X[i]);
                double dz2 = p - y[i];                                 // how wrong the output was
                for (int j = 0; j < W2.Length; j++)
                    dW2[j] += a1[j] * dz2;
                db2 += dz2;
                for (int j = 0; j < hidden; j++)
                {
                    // blame flows back: through W2, then through the tanh's slope
                    double dz1 = dz2 * W2[j] * (bend ? 1 - a1[j] * a1[j] : 1);
                    dW1[0, j] += X[i][0] * dz1;
                    dW1[1, j] += X[i][1] * dz1;
                    db1[j] += dz1;
                }
            }
            for (int j = 0; j < hidden; j++)
            {
                W1[0, j] -= learningRate * dW1[0, j] / n;
                W1[1, j] -= learningRate * dW1[1, j] / n;
                b1[j] -= learningRate * db1[j] / n;
            }
            for (int j = 0; j < W2.Length; j++)
                W2[j] -= learningRate * dW2[j] / n;
            b2 -= learningRate * db2 / n;
        }
    }

    public double Accuracy(double[][] X, double[] y) =>
        100.0 * Enumerable.Range(0, y.Length).Count(i => Forward(X[i]).p > 0.5 == (y[i] == 1)) / y.Length;

    static double Sigmoid(double z) => 1 / (1 + Math.Exp(-z));

    // same as Math.Tanh, written with Exp so Python gets identical digits
    static double Tanh(double z) => 1 - 2 / (Math.Exp(2 * z) + 1);
}

/// <summary>A tiny random generator, written by hand so the Python version
/// starts from exactly the same weights.</summary>
class SplitMix64(ulong state)
{
    public double Uniform()     // a number between -1 and 1
    {
        state += 0x9E3779B97F4A7C15;
        ulong z = state;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EB;
        z ^= z >> 31;
        return (z >> 11) / (double)(1UL << 53) * 2 - 1;
    }
}
