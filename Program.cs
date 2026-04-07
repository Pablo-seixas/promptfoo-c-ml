using Microsoft.ML;
using Microsoft.ML.Data;
using System.Text.Json.Serialization;

var builder = WebApplication.CreateBuilder(args);
var app = builder.Build();

var ml = new MLContext(seed: 42);

var samples = new List<TicketData>
{
    new() { Text = "fatura cobrada em duplicidade no cartao", Label = "billing" },
    new() { Text = "valor incorreto na mensalidade", Label = "billing" },
    new() { Text = "reembolso nao caiu", Label = "billing" },
    new() { Text = "pix cobrado duas vezes", Label = "billing" },
    new() { Text = "nao consigo pagar boleto", Label = "billing" },
    new() { Text = "erro 500 ao abrir dashboard", Label = "technical" },
    new() { Text = "api retorna timeout", Label = "technical" },
    new() { Text = "sistema trava ao salvar pedido", Label = "technical" },
    new() { Text = "falha de conexao com banco", Label = "technical" },
    new() { Text = "endpoint nao responde", Label = "technical" },
    new() { Text = "nao consigo fazer login", Label = "account" },
    new() { Text = "esqueci minha senha", Label = "account" },
    new() { Text = "quero alterar meu email cadastrado", Label = "account" },
    new() { Text = "minha conta foi bloqueada", Label = "account" },
    new() { Text = "nao recebo codigo de verificacao", Label = "account" },
    new() { Text = "suspeita de acesso indevido", Label = "security" },
    new() { Text = "atividade estranha na conta", Label = "security" },
    new() { Text = "token exposto em repositorio", Label = "security" },
    new() { Text = "acho que houve vazamento de dados", Label = "security" },
    new() { Text = "tentativas de login suspeitas", Label = "security" },
    new() { Text = "cobranca errada no plano anual", Label = "billing" },
    new() { Text = "nao consigo resetar a senha", Label = "account" },
    new() { Text = "latencia alta na aplicacao", Label = "technical" },
    new() { Text = "usuario desconhecido acessou meu perfil", Label = "security" }
};

var data = ml.Data.LoadFromEnumerable(samples);
var split = ml.Data.TrainTestSplit(data, testFraction: 0.25);

var pipeline =
    ml.Transforms.Conversion.MapValueToKey("Label")
    .Append(ml.Transforms.Text.FeaturizeText("Features", nameof(TicketData.Text)))
    .Append(ml.MulticlassClassification.Trainers.SdcaMaximumEntropy(labelColumnName: "Label", featureColumnName: "Features"))
    .Append(ml.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

var model = pipeline.Fit(split.TrainSet);
var predictions = model.Transform(split.TestSet);
var metrics = ml.MulticlassClassification.Evaluate(predictions);

app.MapGet("/", () => Results.Json(new
{
    name = "AiApi ML.NET demo",
    status = "online",
    task = "text-classification",
    categories = new[] { "billing", "technical", "account", "security" }
}));

app.MapGet("/metrics", () => Results.Json(new
{
    macroAccuracy = Math.Round(metrics.MacroAccuracy, 4),
    microAccuracy = Math.Round(metrics.MicroAccuracy, 4),
    logLoss = Math.Round(metrics.LogLoss, 4),
    trainedSamples = samples.Count
}));

app.MapPost("/predict", (PredictRequest request) =>
{
    if (string.IsNullOrWhiteSpace(request.Text))
    {
        return Results.BadRequest(new { error = "text is required" });
    }

    var engine = ml.Model.CreatePredictionEngine<TicketData, TicketPrediction>(model);
    var pred = engine.Predict(new TicketData { Text = request.Text });

    var scores = pred.Score ?? Array.Empty<float>();
    var confidence = scores.Length > 0 ? scores.Max() : 0f;

    var answer = $"Categoria prevista: {pred.PredictedLabel}\nConfianca: {confidence:P2}\nResumo: O texto foi classificado como {pred.PredictedLabel}.";

    return Results.Json(new PredictResponse
    {
        Category = pred.PredictedLabel ?? "unknown",
        Confidence = Math.Round(confidence, 4),
        Answer = answer
    });
});

app.Run("http://localhost:5000");

public class TicketData
{
    public string Text { get; set; } = "";
    public string Label { get; set; } = "";
}

public class TicketPrediction
{
    [ColumnName("PredictedLabel")]
    public string PredictedLabel { get; set; } = "";
    public float[] Score { get; set; } = Array.Empty<float>();
}

public class PredictRequest
{
    [JsonPropertyName("text")]
    public string Text { get; set; } = "";
}

public class PredictResponse
{
    [JsonPropertyName("category")]
    public string Category { get; set; } = "";

    [JsonPropertyName("confidence")]
    public double Confidence { get; set; }

    [JsonPropertyName("answer")]
    public string Answer { get; set; } = "";
}
