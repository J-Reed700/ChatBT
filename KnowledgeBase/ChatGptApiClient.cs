using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Net.Http;
using System.Text;
using System.Threading.Tasks;

namespace KnowledgeBase
{
    public class OpenAi
    {
        internal EmbeddingAPI Embedding;
        public OpenAi(string apiKey = "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")
        {
            Embedding = new EmbeddingAPI(apiKey, OpenAIEndpoint.Embeddings);
        }
    }
    internal class EmbeddingAPI : OpenAIBaseClient
    {
        private readonly string _apiKey;
        private readonly string COMPLETIONS_MODEL = "text-davinci-003";
        private readonly string EMBEDDING_MODEL = "text-embedding-ada-002";
        private readonly OpenAIEndpoint _endpoint;
        private readonly string _url;
        private readonly string _embeddingsEndpoint = "https://api.openai.com/v1/embeddings";

        private int maxCharsPerChunk = 10000; // Estimated characters, assuming 5 characters per token


        public EmbeddingAPI(string apiKey, OpenAIEndpoint endpoint)
        {
            _apiKey = apiKey;
            _endpoint = endpoint;
            _url = _embeddingsEndpoint;
        }

 


        public List<string> ChunkText(string text)
        {
            List<string> chunks = new List<string>();

            while (text.Length > 0)
            {
                int charCount = Math.Min(maxCharsPerChunk, text.Length);
                chunks.Add(text.Substring(0, charCount));
                text = text.Substring(charCount);
            }

            return chunks;
        }

        public async Task<List<float>> ProcessRequest(string inputText, string inputString = "")
        {
            try
            {
                using (HttpClient client = new HttpClient())
                {
                    client.Timeout = TimeSpan.FromMinutes(10);
                    client.DefaultRequestHeaders.Add("Authorization", $"Bearer {_apiKey}");

                    var request = Create(inputText, 500);
                    var json = JsonConvert.SerializeObject(request);
                    var contentData = new StringContent(json, Encoding.UTF8, "application/json");
                    var requestBody = new HttpRequestMessage(HttpMethod.Post, new Uri(_url))
                    {
                        Content = contentData
                    };
                    var response = client.SendAsync(requestBody).GetAwaiter().GetResult();

                    if (!response.IsSuccessStatusCode)
                    {
                        var responseContent = await response.Content.ReadAsStringAsync();
                        throw new HttpRequestException($"Error {response.StatusCode}: {response.ReasonPhrase} - {responseContent}");
                    }

                    var responseJson = await response.Content.ReadAsStringAsync();
                    var responseObject = JsonConvert.DeserializeObject<ChatEmbeddingsResponse>(responseJson);

                    if (responseObject == null || responseObject.Data == null || responseObject.Data[0].Embedding == null)
                    {
                        throw new Exception("Error deserializing response JSON from OpenAI.");
                    }
                    return responseObject.Data[0].Embedding;
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.Message);
                return null;
            }
        }

        public override object Create(string textInput, int maxTokens = 500)
        {
            return new
            {
                model = EMBEDDING_MODEL, // Required: ID of the model to use
                input = textInput
            };
        }
    }

    public class ChatResponse
    {
        public string? Id { get; set; }
        public string? Object { get; set; }
        public long Created { get; set; }
        public ChatChoice[]? Choices { get; set; }
        public Usage? Usage { get; set; }
    }

    public class ChatEmbeddingsResponse
    {
        public string? Id { get; set; }
        public string? Object { get; set; }
        public List<Data> Data { get; set;}
        public Usage? Usage { get; set; }
    }

    public class ChatChoice
    {
        public int Index { get; set; }
        public Message? Message { get; set; }
        public string? FinishReason { get; set; }
    }

    public class Message
    {
        public string? Role { get; set; }
        public string? Content { get; set; }
    }

    public class Usage
    {
        public int PromptTokens { get; set; }
        public int CompletionTokens { get; set; }
        public int TotalTokens { get; set; }
    }

    public class Data
    {
        public string? Object { get; set; }
        public int index { get; set;}
        public List<float> Embedding { get; set; }
        public Data()
        {
            Embedding = new List<float>();
        }

    }

    public enum OpenAIEndpoint
    {
        Completions,
        Embeddings
    }
}


//public async Task<string> GenerateCompletion(List<string> messages, ECompletionType completionType)
//{
//    var messageObjects = new List<object>();
//    var systemPrompt = CreateSystemPrompt(completionType);
//    messageObjects.Add(new { role = "system", content = systemPrompt });

//    foreach (string message in messages)
//    {
//        messageObjects.Add(new { role = "user", content = message });
//    }

//    var request = CreateRequest(messageObjects);
//    var json = JsonConvert.SerializeObject(request);
//    var contentData = new StringContent(json, Encoding.UTF8, "application/json");
//    var response = await _client.PostAsync(_url, contentData);

//    if (!response.IsSuccessStatusCode)
//    {
//        throw new HttpRequestException($"Error {response.StatusCode}: {response.ReasonPhrase}");
//    }

//    var responseJson = await response.Content.ReadAsStringAsync();
//    var responseObject = JsonConvert.DeserializeObject<ChatResponse>(responseJson);

//    if (responseObject == null || responseObject.Choices == null || responseObject.Choices[0].Message.Content == null)
//    {
//        throw new Exception("Error deserializing response JSON from OpenAI.");
//    }
//    return responseObject.Choices[0].Message.Content;
//}