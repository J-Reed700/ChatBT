using Microsoft.Data.Analysis;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace KnowledgeBase
{
    public static class DataFrameExtensions
    {
        public static List<DataFrameRow> IterateRows(this DataFrame df)
        {
            return df.Rows.ToList();
        }
    }
}
