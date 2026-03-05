OpenRouter client usage
======================

Configuration (config.yaml):

```yaml
project_name: recallbox
debug: false
embedding_model: "text-embedding-3-small"
chat_model: "gpt-4o-mini"
openrouter_base_url: "https://openrouter.ai/api/v1"
```

Usage example (async):

```py
from recallbox.llm.client import OpenRouterClient

client = OpenRouterClient(api_key="<secret>", embedding_model="text-embedding-3-small", chat_model="gpt-4o-mini")
embs = await client.embed(["hello", "world"])
reply = await client.chat([{"role": "user", "content": "hello"}])
ok, explanation = await client.evaluate_memory("last user message", "last assistant message")
```

Testing: pass a `config` object to the client to avoid global config singleton during tests:

```py
from recallbox.config import get_config
cfg = get_config()
client = OpenRouterClient(api_key=..., embedding_model=..., chat_model=..., config=cfg)
```
