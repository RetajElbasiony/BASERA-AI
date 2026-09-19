
import sounddevice as sd
from scipy.io.wavfile import write

from faster_whisper import WhisperModel
from gtts import gTTS


print("جاري تحميل موديل Whisper...")

# Tiny أسرع وأخف على الـCPU
stt_model = WhisperModel(
    "tiny",
    device="cpu",
    compute_type="int8"
)

print("الموديل جاهز.")


def speech_to_text(audio_file):
    """
    بياخد مسار ملف صوت
    ويرجع (النص، اللغة المكتشفة).
    """

    segments, info = stt_model.transcribe(
        audio_file,
        language="ar",
        beam_size=5
    )

    text = ""

    for segment in segments:
        text += segment.text

    return text.strip(), info.language


def text_to_speech(text, lang="ar", output_file="output.mp3"):
    """
    بياخد نص ولغة، ويحفظه كملف صوت MP3.
    """

    tts = gTTS(
        text=text,
        lang=lang
    )

    tts.save(output_file)

    return output_file


def record_and_transcribe(duration=5, samplerate=44100):
    """
    بتسجل صوت من المايك مباشرة لمدة 5 ثواني،
    وترجع (النص، اللغة).
    """

    filename = "mic_temp.wav"

    print(f"جاري التسجيل لمدة {duration} ثواني...")

    recording = sd.rec(
        int(duration * samplerate),
        samplerate=samplerate,
        channels=1,
        dtype="int16"
    )

    sd.wait()

    write(
        filename,
        samplerate,
        recording
    )

    print("تم التسجيل، جاري التحويل لنص...")

    return speech_to_text(filename)


if __name__ == "__main__":

    # تسجيل الصوت وتحويله إلى Text
    text, lang = record_and_transcribe(duration=5)

    print("النص:", text)
    print("اللغة:", lang)

    # اختبار Text-to-Speech
    text_to_speech(
        "هذا اختبار للصوت في مشروع باسيرا",
        lang="ar",
        output_file="test_output.mp3"
    )

    print("تم إنشاء الصوت بنجاح")
