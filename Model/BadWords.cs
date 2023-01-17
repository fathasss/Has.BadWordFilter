using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;


namespace Has.BadWordsFilter.Model
{
    public class BadWords
    {
        public string Words { get; set; }
    }
    public class BadWordPrediction : BadWords
    {
        [ColumnName("PredictedLabel")]
        public bool Prediction { get; set; }
        public float Probability { get; set; }
    }
}

