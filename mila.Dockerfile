# The RunPod worker, built here rather than on RunPod.
#
# ivrit.ai's image (github.com/ivrit-ai/runpod-serverless) holds only the turbo
# engines, and its worker loads with local_files_only=True, so an engine that
# is not baked in fails on arrival. Building their Dockerfile on RunPod with
# the full engine added did not finish: RunPod's builder stops at 30 minutes,
# and its downloads ran at ~100 kB/s.
#
# Their Dockerfile, with two changes: the engines are the two this server
# offers -- not the Yiddish one, not OpenAI's untuned large-v3-turbo -- and
# their worker script is pinned to a commit, so a rebuild is the same image.
#
# Built and pushed by .github/workflows/mila-image.yml. Source of truth:
# runpod/Dockerfile in the Mila repository (dev-ops/Mila).
FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime

LABEL org.opencontainers.image.source="https://github.com/gute1/runpod-serverless-full"
LABEL org.opencontainers.image.description="ivrit.ai RunPod worker with the turbo and full Hebrew Whisper engines"

WORKDIR /

ENV LD_LIBRARY_PATH="/opt/conda/lib/python3.11/site-packages/nvidia/cudnn/lib:/opt/conda/lib/python3.11/site-packages/nvidia/cublas/lib"

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir ivrit[all]==0.2.6 torch==2.7.1 torchaudio==2.7.1 \
    torchvision==0.22.1 huggingface-hub==0.36.0 runpod

# The engines this server's catalog offers (app/engines.py).
RUN python3 -c 'import faster_whisper; faster_whisper.WhisperModel("ivrit-ai/whisper-large-v3-turbo-ct2", device="cpu")'
RUN python3 -c 'import faster_whisper; faster_whisper.WhisperModel("ivrit-ai/whisper-large-v3-ct2", device="cpu")'

# Speaker detection in the cloud, as in theirs.
RUN python3 -c 'import pyannote.audio; import torch; from pyannote.audio.core.task import Problem, Resolution, Specifications; torch.serialization.add_safe_globals([Problem, Resolution, Specifications, torch.torch_version.TorchVersion]); p = pyannote.audio.Pipeline.from_pretrained("ivrit-ai/pyannote-speaker-diarization-3.1")'
RUN python3 -c 'from speechbrain.inference.speaker import EncoderClassifier; EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")'

ADD https://raw.githubusercontent.com/ivrit-ai/runpod-serverless/fe85652178022e40054f594e52bd1b76695d2b0e/infer.py /infer.py

CMD [ "python", "-u", "/infer.py" ]
