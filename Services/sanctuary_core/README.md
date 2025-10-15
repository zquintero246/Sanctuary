# Sanctuary Core Realtime Pipeline

This package hosts the orchestrator, interface contracts, telemetry tracer, and
auxiliary utilities used by the realtime Sanctuary voice pipeline.

## Quickstart

1. Install dependencies:
   ```bash
   pip install -r Sanctuary/requirements.txt
   ```
2. Start the websocket server:
   ```bash
   bash scripts/run_server.sh
   ```
3. Launch the local microphone client in a second terminal:
   ```bash
   bash scripts/run_client.sh
   ```

With both processes running you can speak into the microphone and hear the
synthesised assistant reply. The client prints `stt_partial`, `stt_final`,
`assistant_text`, and `metrics` events so you can verify latency expectations.

## Environment knobs

The following environment variables can be tweaked before launching the server:

| Variable | Default | Description |
| --- | --- | --- |
| `SANCTUARY_SR` | `16000` | Pipeline sample rate in Hz for input/output audio. |
| `SANCTUARY_FRAME_MS` | `20` | Size of incoming PCM frames in milliseconds. |
| `SANCTUARY_STT_MODEL` | `small` | Whisper checkpoint to use for streaming STT. |
| `SANCTUARY_LLM_MODEL` | `distilgpt2` | HuggingFace model for streaming generation. |
| `SANCTUARY_TTS_MODEL` | `tts_models/multilingual/multi-dataset/xtts_v2` | Coqui XTTS checkpoint. |

## Latency metrics

The orchestrator records key latency marks with `Tracer` and sends them back to
clients once a turn completes:

- `stt_first_partial_ms`
- `stt_final_ms`
- `llm_first_token_ms`
- `tts_first_audio_ms`
- `turn_total_ms`

These metrics are printed by the client and logged on the server to help tune
barge-in behaviour and stage-by-stage latency budgets.
