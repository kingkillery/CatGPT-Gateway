"""Keep the suite hermetic.

src/config.py loads the developer's .env at import. With a bridge model and API key in it,
tests that expect an empty reply to be a 502, or a tool-result turn to reach the fake browser,
would instead call the real bridge model with real credentials. Blank those settings before
anything imports src.config (load_dotenv does not override variables that are already set),
so no test depends on a local .env or spends a real key. Tests that need them set them
explicitly with patch.dict.
"""

import os

for _name in ("OPENROUTER_API_KEY", "HF_TOKEN", "CATGPT_BRIDGE_MODEL"):
    os.environ[_name] = ""
