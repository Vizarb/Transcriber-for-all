from pathlib import Path
import tempfile
import asyncio
from functools import partial
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from transcription_service import transcribe

app = FastAPI(title="Whisper Transcription Service")

async def run_transcription_thread(fn, *args, **kwargs):
    """
    Execute a blocking function in a thread pool and return its result.
    Using functools.partial here avoids the need for an anonymous lambda.
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, partial(fn, *args, **kwargs))

@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    """
    Serve a simple HTML page with a drag-and-drop upload form.
    """
    html = """
    <!DOCTYPE html>
    <html>
    <head>
      <meta charset="utf-8">
      <title>Transcription Service</title>
      <style>
        #drop_zone {
          border: 3px dashed #888;
          padding: 1rem;
          text-align: center;
          cursor: pointer;
        }
        #drop_zone.highlight { background: #f0f8ff; }
      </style>
    </head>
    <body>
      <h2>Upload a file to transcribe</h2>
      <form action="/transcribe" method="post" enctype="multipart/form-data">
        <div id="drop_zone">Drag &amp; drop a file here or click to select</div>
        <input type="file" id="file_input" name="file" style="display:none">
        <p>
          Model:
          <select name="model">
            <option value="tiny">tiny</option>
            <option value="base">base</option>
            <option value="small" selected>small</option>
            <option value="medium">medium</option>
            <option value="large-v3">large-v3</option>
          </select>
        </p>
        <p>Language (blank for auto): <input type="text" name="lang"></p>
        <p>Temperature: <input type="number" name="temp" step="0.1" value="0.0"></p>
        <p>Chunk length (sec): <input type="number" name="chunk" value="30"></p>
        <p>Beam size: <input type="number" name="beam" value="5"></p>
        <p>Initial prompt: <input type="text" name="prompt"></p>
        <p>
          <label><input type="checkbox" name="timestamps"> Include timestamps</label><br>
          <label><input type="checkbox" name="srt"> Generate SRT file</label><br>
          <label><input type="checkbox" name="translate"> Translate to English</label><br>
          <label><input type="checkbox" name="no_vad"> Disable VAD</label><br>
          <label><input type="checkbox" name="noprev"> Do not condition on previous text</label><br>
          <label><input type="checkbox" name="song"> Extract vocals (Demucs)</label><br>
          Demucs model: <input type="text" name="demucs_model" value="htdemucs">
        </p>
        <button type="submit">Upload &amp; Transcribe</button>
      </form>

      <script>
        const dropZone = document.getElementById('drop_zone');
        const fileInput = document.getElementById('file_input');

        dropZone.addEventListener('dragover', (ev) => {
          ev.preventDefault();
          dropZone.classList.add('highlight');
        });
        dropZone.addEventListener('dragleave', () => {
          dropZone.classList.remove('highlight');
        });
        dropZone.addEventListener('drop', (ev) => {
          ev.preventDefault();
          dropZone.classList.remove('highlight');
          if (ev.dataTransfer.files.length > 0) {
            fileInput.files = ev.dataTransfer.files;
          }
        });
        dropZone.addEventListener('click', () => fileInput.click());
      </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html)

@app.post("/transcribe")
async def transcribe_endpoint(
    file: UploadFile = File(...),
    model: str = Form("small"),
    lang: str | None = Form(None),
    temp: float = Form(0.0),
    chunk: int = Form(30),
    timestamps: str | None = Form(None),
    srt: str | None = Form(None),
    translate: str | None = Form(None),
    no_vad: str | None = Form(None),
    beam: int = Form(5),
    noprev: str | None = Form(None),
    prompt: str | None = Form(None),
    song: str | None = Form(None),
    demucs_model: str = Form("htdemucs"),
) -> FileResponse:
    """
    Receive the uploaded file, run transcription, and return the output as a download.
    """
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")
    if lang == "":
        lang = None

    # Convert checkbox values ("on"/None) to booleans
    to_bool = lambda v: bool(v and v.lower() == "on")
    timestamps_flag = to_bool(timestamps)
    srt_flag = to_bool(srt)
    translate_flag = to_bool(translate)
    no_vad_flag = to_bool(no_vad)
    noprev_flag = to_bool(noprev)
    song_flag = to_bool(song)

    contents = await file.read()
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = Path(tmpdir) / file.filename
        with open(input_path, "wb") as f_out:
            f_out.write(contents)

        result = await run_transcription_thread(
            transcribe,
            media_path=input_path,
            model=model,
            lang=lang,
            temp=temp,
            chunk=chunk,
            timestamps=timestamps_flag,
            srt=srt_flag,
            translate=translate_flag,
            no_vad=no_vad_flag,
            beam=beam,
            noprev=noprev_flag,
            prompt=prompt,
            song=song_flag,
            demucs_model=demucs_model,
        )

        txt_path = result["txt_path"]
        if srt_flag and "srt_path" in result:
            srt_path = result["srt_path"]
            import zipfile
            zip_path = txt_path.parent / f"{txt_path.stem}.zip"
            with zipfile.ZipFile(zip_path, "w") as zf:
                zf.write(txt_path, txt_path.name)
                zf.write(srt_path, srt_path.name)
            return FileResponse(zip_path, filename=zip_path.name)

        return FileResponse(txt_path, filename=txt_path.name)
