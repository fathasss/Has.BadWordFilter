using Has.BadWordsFilter.Model;
using Microsoft.ML;
using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Has.BadWordsFilter
{
    public class ProfanityDetector
    {
        private PredictionEngine<BadWords, BadWordPrediction> _predictionEngine;

        public ProfanityDetector() { LoadTrainedModel(); }

        private void LoadTrainedModel()
        {
            DataViewSchema modelSchema;
            var mlContext = new MLContext();
            var trainedModel = mlContext.Model.Load(GetModelStream(), out modelSchema);
            _predictionEngine = mlContext.Model.CreatePredictionEngine<BadWords, BadWordPrediction>(trainedModel);
        }

        public Stream GetModelStream()
        {
            CreateModel.ModelTrainer();
            string modelPath = File.ReadAllText("Has.SadFilter/Data/model.zip");
            var assembly = typeof(ProfanityDetector).Assembly;
            return assembly.GetManifestResourceStream(modelPath);
        }

        public bool IsProfane(string word)
        {
            var obj = new BadWords { Words = word };
            return _predictionEngine.Predict(obj).Prediction;
        }

        public float GetProfanityProbability(string word)
        {
            var obj = new BadWords { Words = word };
            return _predictionEngine.Predict(obj).Probability;
        }
    }
}

