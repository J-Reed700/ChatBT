using AI.Dev.OpenAI.GPT;
using Microsoft.Data.Analysis;
using NumSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using System.Xml.Linq;

namespace KnowledgeBase
{
    public class EmbeddingsGenerator
    {
        private OpenAi openai;
        private const string SEPARATOR = "\n* ";
        private const string ENCODING = "gpt2"; // encoding for text-davinci-003
        private const int MAX_SECTION_LEN = 2048; // maximum length of the selected document sections, in tokens
        private readonly string DATA_FILE;
        private readonly string EMBED_FILE;
        private Dictionary<Tuple<string, string>, List<float>> _embeddings;
        private DataFrame df;


        public EmbeddingsGenerator(string dataFile, string embeddingFile = "") { 
            DATA_FILE = dataFile;
            EMBED_FILE = embeddingFile;           
            openai = new OpenAi();
            Init();
        }

        private async void Init()
        {
            _embeddings = await LoadEmbeddings(!string.IsNullOrEmpty(EMBED_FILE));
        }

        private async Task<List<float>> GetEmbedding(string text)
        {
            var result = await openai.Embedding.ProcessRequest(text);
            return result;
        }
  
        private async Task<Dictionary<Tuple<string, string>, List<float>>> ComputeDocEmbeddings(DataFrame df)
        {
            var returnDict = new Dictionary<Tuple<string,string>, List<float>>();
            foreach(DataFrameRow row in df.IterateRows())
            {
                if (CountTokens(row[2].ToString()) > 8192)
                {
                    var chunks = ChunkText(row[2].ToString());
                    foreach (var chunk in chunks)
                    {
                        returnDict[new(row[0].ToString(), row[1].ToString())] = await this.GetEmbedding(chunk);
                    }
                }
                else
                {
                    returnDict[new(row[0].ToString(), row[1].ToString())] = await this.GetEmbedding(row[2].ToString());
                }
            }
            return returnDict;
        }

        // 
        //     Read the document embeddings and their keys from a CSV.
        //     
        //     fname is the path to a CSV with exactly these named columns: 
        //         "title", "heading", "0", "1", ... up to the length of the embedding vectors.
        //     
        private async Task<Dictionary<Tuple<string, string>, List<float>>> LoadEmbeddings(bool alreadyCreated = false)
        {
            try
            {
                df = DataFrame.LoadCsv(DATA_FILE);
                var embeddingDict = new Dictionary<Tuple<string, string>, List<float>>();
                var maxDim = 0;
                if (alreadyCreated)
                { 
                    /*df = DataFrame.LoadCsv(@"C:\Users\KBEmbeddingsv2.csv");
                    foreach (DataFrameRow row in df.IterateRows())
                    {
                        if (maxDim == 0)
                        {
                            maxDim = row.Skip(2).Count();
                        }
                        var title = row[0].ToString();
                        var heading = row[1].ToString();
                        var embedding = new List<float>();
                        for (int i = 0; i <= maxDim; i++)
                        {
                            embedding.Add(float.Parse(row[i + 2].ToString()));
                        }
                        embeddingDict.Add(new(title, heading), embedding);
                    }*/
                    using (var reader = new StreamReader(@"C:\Users\KBEmbeddingsv2.csv"))
                    {
                        var header = reader.ReadLine(); // skip the header line
                        maxDim = header.Split(",").Count() - 2; // the number of dimensions in each embedding vector
                        while (!reader.EndOfStream)
                        {
                            var line = reader.ReadLine();
                            Regex CSVParser = new Regex(",(?=(?:[^\"]*\"[^\"]*\")*(?![^\"]*\"))");
                            var values = CSVParser.Split(line);
                            var title = values[0];
                            var heading = values[1];
                            var embedding = new List<float>();
                            for (int i = 0; i < maxDim; i++)
                            {
                                float flt;
                                if (float.TryParse(values[i + 2], out flt))
                                {
                                    embedding.Add(float.Parse(values[i + 2]));
                                }
                                else
                                {
                                    var x = values[i + 2];
                                }
                            }
                            embeddingDict.Add(new (title, heading), embedding);
                        }
                    }

                    return embeddingDict;
                }
                else
                {
                    embeddingDict = await ComputeDocEmbeddings(df);
                    WriteToCsv();
                }
                return embeddingDict;
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.ToString());
                return null;
            }
        }

        public NDArray VectorSimilarity(List<float> x, List<float> y)
        {
            return np.dot(np.array(x), np.array(y));
        }

        private static int CountTokens(string text)
        {
            return GPT3Tokenizer.Encode(text).Count;
        }
        public List<string> ChunkText(string text)
        {
            List<string> chunks = new List<string>();

            while (text.Length > 0)
            {
                int charCount = Math.Min(8192, text.Length);
                chunks.Add(text.Substring(0, charCount));
                text = text.Substring(charCount);
            }

            return chunks;
        }

        public void WriteToCsv(string filename = @"C:\Users\KBEmbeddings.csv")
        {
            if (_embeddings is not null && _embeddings.Count() > 0)
            {
                using (var writer = new StreamWriter(filename))
                {
                    writer.WriteLine("title,heading," + String.Join(",", Enumerable.Range(0, _embeddings.FirstOrDefault().Value.Count())));
                    foreach (var embedding in _embeddings)
                    {
                        writer.WriteLine("{0},{1},{2}", EscapeCsv(embedding.Key.Item1.ToString()), EscapeCsv(embedding.Key.Item2.ToString()), String.Join(";", embedding.Value));
                    }
                }
            }
        }


        public async Task<List<(NDArray, Tuple<string, string>)>> OrderDocumentSectionsByQuerySimilarity(string query)
        {
            try
            {
                /*
                 * Find the query embedding for the supplied query, and compare it against all of the pre-calculated document embeddings
                 * to find the most relevant sections.
                 *
                 * Return the list of document sections, sorted by relevance in descending order.
                 */
                var queryEmbedding = await this.GetEmbedding(query);
                var documentSimilarities = _embeddings.Select(context =>
                    (VectorSimilarity(queryEmbedding.ToList(), context.Value.ToList()), context.Key))
                    .OrderByDescending(item => item.Item1.GetAtIndex(0))
                    .ToList();

                return documentSimilarities;
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.Message);
                return null;
            }
        }


        public void FixCsvFile()
        {
            using (var reader = new StreamReader(@"C:\Users\KBEmbeddings.csv"))
            {
                using (var writer = new StreamWriter(@"C:\Users\KBEmbeddingsv2.csv"))
                {
                    writer.WriteLine("title,heading," + String.Join(",", Enumerable.Range(0, 1535)));
                    while(!reader.EndOfStream) { 
                        var rows = reader.ReadLine().Replace("&nbsp;","").Replace("&amp;","").Split(";").ToList();
                        writer.WriteLine("{0},{1},{2}", EscapeCsv(rows[0]), EscapeCsv(rows[1]), String.Join(",", rows.Skip(2)));
                    }
                }
            }
        }

        public async Task<string> ConstructPrompt(string question)
        {
            /*
             * Fetch relevant document sections based on the given question and construct a prompt
             * for the OpenAI GPT-3 API.
             * contextEmbeddings: a dictionary mapping (title, heading) pairs to embedding vectors.
             * df: a pandas DataFrame containing the document contents and their lengths in tokens.
             */
            var mostRelevantDocumentSections = await OrderDocumentSectionsByQuerySimilarity(question);

            var chosenSections = new List<string>();
            var chosenSectionsLen = 0;
            var chosenSectionsIndexes = new List<string>();

            foreach (var (_, sectionIndex) in mostRelevantDocumentSections)
            {
                // Add contexts until we run out of space.
                var documentSection = df.Rows.FirstOrDefault(x => x[1].ToString().Contains(sectionIndex.Item2));

                chosenSectionsLen += Convert.ToInt32(documentSection[3].ToString());
                if (chosenSectionsLen > MAX_SECTION_LEN)
                {
                    break;
                }

                var content = documentSection[2].ToString();
                chosenSections.Add(SEPARATOR + content);
                chosenSectionsIndexes.Add(sectionIndex.ToString());
            }

            // Useful diagnostic information.
            Console.WriteLine($"Selected {chosenSections.Count} document sections:");
            Console.WriteLine(string.Join("\n", chosenSectionsIndexes));

            var header = @"Answer the question as truthfully as possible using the provided context, and if the answer is not contained within the text below, say ""I don't know."" Context:";
            return header + string.Join("", chosenSections) + $"\n\nQ: {question}\nA:";
        }

        private string EscapeCsv(string data)
        {
            var newRow = data;
            if (data.Contains("\""))
            {
                newRow = data.Replace("\"", "\"\"");
                newRow = String.Format("\"{0}\"", newRow);
            }
            else if (data.Contains(",") || data.Contains(System.Environment.NewLine))
            {
                newRow = String.Format("\"{0}\"", data);
            }
            return newRow;
        }
    }
}
