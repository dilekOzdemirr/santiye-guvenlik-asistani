import json
import os
from datetime import datetime, timezone
from urllib import request
from urllib.error import URLError


SAFETY_API_URL = os.getenv("SAFETY_API_URL", "http://localhost:4000/api/violations")


def ihlal_gonder(
    ihlal_turu,
    fotograf_yolu,
    baret_var_mi,
    yelek_var_mi,
    bbox=None,
    kaynak="opencv",
    tehlikeli_bolge=False,
    video_job_id=None,
    confidence=None,
):
    payload = {
        "violationType": ihlal_turu,
        "photoPath": fotograf_yolu,
        "helmetDetected": baret_var_mi,
        "vestDetected": yelek_var_mi,
        "dangerZone": tehlikeli_bolge,
        "bbox": bbox,
        "source": kaynak,
        "detectedAt": datetime.now(timezone.utc).isoformat(),
    }

    if video_job_id is not None:
        payload["videoJobId"] = video_job_id

    if confidence is not None:
        payload["confidence"] = confidence

    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    api_request = request.Request(
        SAFETY_API_URL,
        data=data,
        method="POST",
        headers={"Content-Type": "application/json"},
    )

    try:
        with request.urlopen(api_request, timeout=3) as response:
            if response.status not in (200, 201):
                print(f"API beklenmeyen yanit verdi: HTTP {response.status}")
                return False
    except URLError as error:
        print(f"API bildirimi gonderilemedi: {error}")
        return False

    return True
