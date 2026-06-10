import numpy as np
import scipy.signal as signal
import sounddevice as sd
import threading
import queue
from rtlsdr import RtlSdr

# ── stałe ────────────────────────────────────────────────────
SDR_FS = 2_400_000
RF_DEC = 10           # 2.4 MHz → 240 kHz
AUDIO_DEC = 5            # 240 kHz → 48 kHz
IF_FS = SDR_FS // RF_DEC               # 240 000 Hz
AUDIO_FS = IF_FS // AUDIO_DEC             # 48 000 Hz
BLOCK_SIZE = 2048
IQ_BLOCK = 64 * 1024
PRE_BUF = AUDIO_FS // 2

# RF LP: ±100 kHz (pełny kanał FM), 128 tapów → dobra selekcja sąsiednich stacji
_RF_LP = signal.firwin(128, 100e3 / (SDR_FS / 2))
# Audio LP: 15 kHz – usuwa pilot 19 kHz i subcarrier stereo 23-53 kHz
_AUDIO_LP = signal.firwin(256, 15e3 / (IF_FS / 2))
# Stała skala FM: odchyłka ±75 kHz → ±1.0 (brak AGC pumping)
_FM_SCALE = float(IF_FS / (2 * np.pi * 75e3))

# ── wspólny bufor audio ────────────────────────────────────────
_buf = np.array([], dtype=np.float32)
_buf_lock = threading.Lock()

# ── kolejka IQ (SDR → demodulator) ────────────────────────────
_iq_q = queue.Queue(maxsize=8)
_stop = threading.Event()


def _audio_cb(outdata, frames, time_info, status):
    global _buf
    with _buf_lock:
        have = len(_buf)
        if have >= frames:
            outdata[:, 0] = _buf[:frames]
            _buf = _buf[frames:]
        else:
            outdata[:] = 0.0
            if have:
                outdata[:have, 0] = _buf
            _buf = np.array([], dtype=np.float32)


# ── demodulator FM ────────────────────────────────────────────
class FMDemodulator:
    def __init__(self):
        tau = 75e-6
        alpha = np.exp(-1.0 / (AUDIO_FS * tau))
        self._b = np.array([1.0 - alpha])
        self._a = np.array([1.0, -alpha])
        self._zi_de = np.zeros(1)
        self._zi_rf_i = np.zeros(len(_RF_LP) - 1)
        self._zi_rf_q = np.zeros(len(_RF_LP) - 1)
        self._zi_au = np.zeros(len(_AUDIO_LP) - 1)
        self._last_iq = np.complex64(1.0)  # stan dyskryminatora między blokami

    def demod(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.complex64)

        # 1. LP + decymacja RF:  2.4 MHz → 240 kHz
        xi, self._zi_rf_i = signal.lfilter(
            _RF_LP, 1.0, x.real.astype(np.float64), zi=self._zi_rf_i)
        xq, self._zi_rf_q = signal.lfilter(
            _RF_LP, 1.0, x.imag.astype(np.float64), zi=self._zi_rf_q)
        x = (xi[::RF_DEC] + 1j * xq[::RF_DEC]).astype(np.complex64)

        # 2. Dyskryminator FM – ciągły między blokami (brak glitcha na granicy)
        x_ext = np.empty(len(x) + 1, dtype=np.complex64)
        x_ext[0] = self._last_iq
        x_ext[1:] = x
        self._last_iq = x[-1]
        disc = np.angle(x_ext[1:] * np.conj(x_ext[:-1])) * _FM_SCALE

        # 3. LP 15 kHz + decymacja audio:  240 kHz → 48 kHz
        disc, self._zi_au = signal.lfilter(
            _AUDIO_LP, 1.0, disc, zi=self._zi_au)
        disc = disc[::AUDIO_DEC]

        # 4. De-emfaza 75 µs
        disc, self._zi_de = signal.lfilter(
            self._b, self._a, disc, zi=self._zi_de)

        # 5. Stała głośność + soft clip (brak AGC pumping)
        return np.clip(disc * 0.8, -1.0, 1.0).astype(np.float32)


# ── callback RTL-SDR (wątek USB, minimum pracy) ───────────────
def _sdr_cb(samples, rtlsdr_obj):
    try:
        _iq_q.put_nowait(np.asarray(samples, dtype=np.complex64).copy())
    except queue.Full:
        pass  # upuszczamy blok zamiast blokować pętlę USB


# ── wątek demodulatora ────────────────────────────────────────
def _demod_worker(fm: FMDemodulator):
    global _buf
    max_buf = AUDIO_FS * 3
    while not _stop.is_set():
        try:
            iq = _iq_q.get(timeout=1.0)
        except queue.Empty:
            continue
        audio = fm.demod(iq)
        with _buf_lock:
            _buf = np.concatenate((_buf, audio))
            if len(_buf) > max_buf:
                _buf = _buf[-max_buf:]


# ── main ──────────────────────────────────────────────────────
def main():
    sdr = RtlSdr(RtlSdr.get_device_index_by_serial('00000001'))
    sdr.sample_rate = SDR_FS
    sdr.center_freq = 103.2e6
    sdr.freq_correction = 77
    sdr.gain = 'auto'

    fm = FMDemodulator()

    # Wątek 1: asynchroniczny odczyt USB → kolejka IQ
    sdr_thread = threading.Thread(
        target=sdr.read_samples_async,
        args=(_sdr_cb,),
        kwargs={'num_samples': IQ_BLOCK},
        daemon=True,
    )
    sdr_thread.start()

    # Wątek 2: demodulacja IQ → bufor audio
    demod_thread = threading.Thread(
        target=_demod_worker, args=(fm,), daemon=True)
    demod_thread.start()

    # Czekaj na wypełnienie pre-bufora przed startem strumienia
    print("Buffering...", end='', flush=True)
    while True:
        with _buf_lock:
            ready = len(_buf) >= PRE_BUF
        if ready:
            break
        sd.sleep(50)
    print(" OK")

    stream = sd.OutputStream(
        samplerate=AUDIO_FS,
        channels=1,
        dtype='float32',
        callback=_audio_cb,
        blocksize=BLOCK_SIZE,
    )
    stream.start()
    print(
        f"FM playing – {sdr.center_freq/1e6:.1f} MHz  |  {AUDIO_FS} Hz  – Ctrl+C to stop")

    try:
        while True:
            sd.sleep(1000)
    except KeyboardInterrupt:
        pass

    _stop.set()
    try:
        sdr.cancel_read_async()
    except Exception:
        pass
    stream.stop()
    stream.close()
    sdr.close()


if __name__ == "__main__":
    main()
