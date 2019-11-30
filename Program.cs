using System;
using System.Collections.Generic;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;

namespace NorthwindDataMining
{
    public class Program
    {
        private const string TrainingDataRelativePath = @"../../../Data/NorthwindData.csv";
        private static readonly string TrainingDataLocation = Helpers.GetAbsolutePath(TrainingDataRelativePath);
        
        private static void Main(string[] args)
        {
            //Passo 1: Criar o contexto do ML
            var mlContext = new MLContext();
            
            //Passo 2: Carregar os dados
            var data = mlContext.Data.LoadFromTextFile(path:TrainingDataLocation,
                new[]
                {
                    new TextLoader.Column("Label", DataKind.Single, 0),
                    new TextLoader.Column(name:nameof(ProductEntry.ProductId), dataKind:DataKind.UInt32, source: new [] { new TextLoader.Range(0) }, keyCount: new KeyCount(77)), 
                    new TextLoader.Column(name:nameof(ProductEntry.CoPurchaseProductId), dataKind:DataKind.UInt32, source: new [] { new TextLoader.Range(1) }, keyCount: new KeyCount(77))
                },
                hasHeader: true,
                separatorChar: ',');
            
            //Passo 3: Construir os parametros do algoritmo de Fatoriazação por matriz
            var options = new MatrixFactorizationTrainer.Options
            {
                MatrixColumnIndexColumnName = nameof(ProductEntry.ProductId),
                MatrixRowIndexColumnName = nameof(ProductEntry.CoPurchaseProductId),
                LabelColumnName = "Label",
                LossFunction = MatrixFactorizationTrainer.LossFunctionType.SquareLossOneClass,
                Alpha = 0.01,
                Lambda = 0.025,
                ApproximationRank = 100,
                C = 0.00001
            };
            
            //Passo 4: Passar os parametros para o algoritmo
            var est = mlContext.Recommendation().Trainers.MatrixFactorization(options);
            
            //Passo 5: Treinar o Modelo com o algoritmo
            ITransformer model = est.Fit(data);

            //Passo 6: Testes com as predições 
            var predictionEngine = mlContext.Model.CreatePredictionEngine<ProductEntry, CoPurchasePrediction>(model);
            var tests = new List<ProductEntry>()
            {
                new ProductEntry()
                {
                    ProductId = 2,
                    CoPurchaseProductId = 62 
                },
                new ProductEntry()
                {
                    ProductId = 3,
                    CoPurchaseProductId = 63
                },
            };

            foreach (var test in tests)
            {
                var prediction = predictionEngine.Predict(test);
                Console.WriteLine($@"\n For ProductId = {test.ProductId} and  CoPurchaseProductID = {test.CoPurchaseProductId} the predicted score is {Math.Round(prediction.Score, 1)}");
            }
            Console.WriteLine("=============== End of process, hit any key to finish ===============");
            Console.ReadKey();
        }

        public class CoPurchasePrediction
        {
            public float Score { get; set; }
        }
        
        public class ProductEntry
        {
            [KeyType(count : 77)]
            public uint ProductId { get; set; }

            [KeyType(count : 77)]
            public uint CoPurchaseProductId { get; set; }
        }
    }
}