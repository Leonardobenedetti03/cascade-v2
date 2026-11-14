"""Gate della cascata: decide, senza guardare pixel, se scalare una clip a L1.

Protocollo (sezione 6 del piano):
- il gate si addestra **solo su uscite fuori campione di L0** delle take di train
  (le predizioni in-sample del train non vanno mai usate);
- score fuori campione sul train con una CV di secondo livello sulle stesse fold;
- fit finale sul train intero; su val si sceglie il punto di lavoro; il test solo
  con ``--final``.
"""

from __future__ import annotations

FIT_SPLIT = "train"
FIT_SOURCE = "oof"


def check_fit_split(split: str, source: str) -> None:
    """Errore se i dati di fit del gate non sono uscite OOF di L0 sullo split di train.

    Args:
        split: split delle righe usate per il fit (deve essere ``"train"``).
        source: provenienza delle uscite di L0 su quelle righe (deve essere ``"oof"``,
            cioè ogni clip predetta da un modello che non l'ha vista in training).
    """
    if split != FIT_SPLIT:
        raise ValueError(f"il gate si addestra solo sullo split {FIT_SPLIT!r}, ricevuto {split!r}")
    if source != FIT_SOURCE:
        raise ValueError(
            f"il gate si addestra solo su uscite fuori campione di L0 (source={FIT_SOURCE!r}), " f"ricevuto {source!r}"
        )
