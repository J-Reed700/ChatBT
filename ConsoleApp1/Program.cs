// See https://aka.ms/new-console-template for more information
using KnowledgeBase;

Console.WriteLine("Hello, World!");
var embeddings = new EmbeddingsGenerator(@"C:\Users\KBResultsV2.csv", @"C:\Users\KBEmbeddingsv2.csv");
var prompt = await embeddings.ConstructPrompt("What's a strong password in Bigtime?");
Console.WriteLine(prompt);