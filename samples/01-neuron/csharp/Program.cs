// Post 1 -- a single neuron (logistic regression) tells Adelie from Chinstrap
// penguins using two numbers: bill length and bill depth.
//
//     dotnet run

using System.Globalization;

// walk up from the build folder until we find the repo's data/ folder
var dir = new DirectoryInfo(AppContext.BaseDirectory);
while (!File.Exists(Path.Combine(dir.FullName, "data", "penguins.csv")))
    dir = dir.Parent ?? throw new FileNotFoundException("data/penguins.csv not found");
var dataPath = Path.Combine(dir.FullName, "data", "penguins.csv");

// --- load: Adelie (0) vs Chinstrap (1), rows with missing bills dropped ----
var rows = File.ReadLines(dataPath).Skip(1)
    .Select(line => line.Split(','))
    .Where(c => (c[0] == "Adelie" || c[0] == "Chinstrap") && c[2] != "NA")
    .ToList();
double[][] X = rows.Select(c => new[] { Parse(c[2]), Parse(c[3]) }).ToArray();
double[] y = rows.Select(c => c[0] == "Chinstrap" ? 1.0 : 0.0).ToArray();

// every 4th penguin is held out for testing; no randomness, so Python matches exactly
var trainIdx = Enumerable.Range(0, y.Length).Where(i => i % 4 != 3).ToArray();
var testIdx = Enumerable.Range(0, y.Length).Where(i => i % 4 == 3).ToArray();

// --- standardize: centre each feature on 0 with spread 1 (train stats only) ----
var mean = new double[2];
var std = new double[2];
for (int j = 0; j < 2; j++)
{
    mean[j] = trainIdx.Average(i => X[i][j]);
    std[j] = Math.Sqrt(trainIdx.Average(i => Math.Pow(X[i][j] - mean[j], 2)));
}
double[][] Xs = X.Select(x => new[] { (x[0] - mean[0]) / std[0], (x[1] - mean[1]) / std[1] }).ToArray();

// --- train: gradient descent on the log loss -----------------------------
var w = new double[2];
double b = 0.0;
const double learningRate = 0.1;
int n = trainIdx.Length;

for (int step = 0; step <= 1000; step++)
{
    double loss = 0, dw0 = 0, dw1 = 0, db = 0;
    foreach (int i in trainIdx)
    {
        double p = Predict(Xs[i]);                  // predicted chance of Chinstrap
        loss -= y[i] * Math.Log(p) + (1 - y[i]) * Math.Log(1 - p);
        double dz = p - y[i];                       // how wrong this prediction was
        dw0 += dz * Xs[i][0];
        dw1 += dz * Xs[i][1];
        db += dz;
    }
    w[0] -= learningRate * dw0 / n;
    w[1] -= learningRate * dw1 / n;
    b -= learningRate * db / n;
    if (step % 200 == 0)
        Console.WriteLine($"step {step,4}  loss {loss / n:F4}");
}

Console.WriteLine($"\nweights: bill length {w[0]:+0.000;-0.000}, bill depth {w[1]:+0.000;-0.000}, bias {b:+0.000;-0.000}");
Console.WriteLine($"train accuracy {Accuracy(trainIdx):F1}%  ({trainIdx.Length} penguins)");
Console.WriteLine($"test  accuracy {Accuracy(testIdx):F1}%  ({testIdx.Length} penguins)");
Console.WriteLine($"always guessing Adelie: {100.0 * testIdx.Count(i => y[i] == 0) / testIdx.Length:F1}%");

double Sigmoid(double z) => 1 / (1 + Math.Exp(-z));
double Predict(double[] x) => Sigmoid(w[0] * x[0] + w[1] * x[1] + b);
double Accuracy(int[] idx) => 100.0 * idx.Count(i => Predict(Xs[i]) > 0.5 == (y[i] == 1)) / idx.Length;
static double Parse(string s) => double.Parse(s, CultureInfo.InvariantCulture);
