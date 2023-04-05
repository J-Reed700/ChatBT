using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace KnowledgeBase
{
    internal abstract class OpenAIBaseClient
    {
        public abstract object Create(string inputText, int maxTokens);
    }
}
