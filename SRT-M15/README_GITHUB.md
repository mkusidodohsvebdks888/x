# SRT-M15 GitHub merge notes

This folder contains the **local bilingual learning** code, merged with earlier
Colab, authenticated-API and optional DeepSeek integrations. Default inference
runs exclusively on local SRT-M15 checkpoints; external providers are opt-in.

## Windows/macOS/Linux local chat & training

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python start_local.py
```

Open <http://127.0.0.1:8765/> for the Chinese/English local learning UI.

## Optional existing integrations preserved in GitHub

- `SRT_M15_Phone_Colab.ipynb`, `chat_ui.py`, `train_qa.py`: earlier Colab
  utilities retained in their original paths. The Notebook embeds its own
  earlier source; it is **not** auto-updated to the new local bilingual server.
- `api_server.py`, `requirements_api.txt`: existing authenticated, limited
  Chat Completions-style API. To run that optional server install both
  `requirements.txt` and `requirements_api.txt`, set a strong `SRT_API_KEY`
  in an environment variable, then run `uvicorn api_server:app --host 127.0.0.1`.
  Do not expose the server to the internet without reviewing its security.
- `WEB_DEEPSEEK.md`, `deepseek_web_cli.py`, `requirements_web.txt`,
  `srt_m15/deepseek_web.py`, `srt_m15/deepseek_api.py`: prior optional
  DeepSeek browser/API providers, preserved. Do not commit API credentials.
- `srt_m15/chat.py`: new bilingual local chat engine plus opt-in optional
  `backend="web"`, `backend="api"`, or `backend="auto"` Python modes.
  The browser UI defaults to `backend="local"` without calling external LLMs.

## Model training safeguards

- Both original `.pt` research checkpoints are preserved and versioned.
- Language training uses approved Q&A, grouping related translations so
  train/validation splitting does not leak paired questions.
- Both languages are independently checked before accepting candidate weights.
- Internet snippets are held in a reviewable knowledge store, never directly
  promoted into trusted training labels.
- Runtime databases, candidate checkpoints, browser profiles and private
  keys are excluded from Git.

This is a small experimental neural model. Improved validation loss on a small
exercise does **not** prove general Chinese/English language understanding.
