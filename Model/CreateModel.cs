using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Has.BadWordsFilter.Model
{
    public class CreateModel
    {
        public static string modelPath = File.ReadAllText(@"C:\Users\fatihh\source\repos\Has.BadWordsFilter\Data\bad-words.json");
        public CreateModel() { }
        public static void ModelTrainer()
        {
            MLContext mlContext = new MLContext();

            var modelJson = JsonConvert.DeserializeObject<List<string>>(modelPath);
            IDataView data = mlContext.Data.LoadFromEnumerable<string>(modelJson);

            EstimatorChain<RegressionPredictionTransformer<LinearRegressionModelParameters>> pipelineEstimator =
                        mlContext.Transforms.Concatenate("Features", new string[] { "Words" })
                            .Append(mlContext.Transforms.NormalizeMinMax("Features"))
                            .Append(mlContext.Regression.Trainers.Sdca());

            ITransformer trainedModel = pipelineEstimator.Fit(data);
            mlContext.Model.Save(trainedModel, data.Schema, "Data/model.zip");
        }
    }
}
