import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BASE = "https://datasets-server.huggingface.co"
DATASET = "clapAI/MultiLingualSentiment"

retry = Retry(
    total=2,
    connect=2,
    read=2,
    status=2,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=frozenset(["GET"]),
    raise_on_status=False,
)

session = requests.Session()
session.mount("https://", HTTPAdapter(max_retries=retry))


def check(name, endpoint, params):
    print(f"\n=== {name} ===")
    try:
        response = session.get(
            f"{BASE}/{endpoint}",
            params=params,
            timeout=(10, 20),
        )
        print("HTTP status:", response.status_code)

        if not response.ok:
            print("Pesan:", response.text[:300])
            return

        data = response.json()

        if endpoint == "is-valid":
            print("Status dataset:", data)
        elif endpoint == "first-rows":
            print("Kolom:", [
                item["name"]
                for item in data.get("features", [])
            ])
            print("Jumlah baris preview:", len(data.get("rows", [])))
        elif endpoint == "filter":
            print("Jumlah baris cocok:", data.get("num_rows_total"))
            for item in data.get("rows", []):
                row = item.get("row", {})
                print({
                    "language": row.get("language"),
                    "label": row.get("label"),
                    "text_preview": str(row.get("text", ""))[:100],
                })

    except requests.RequestException as exc:
        print("Permintaan gagal:", type(exc).__name__, str(exc)[:250])


check("Validasi dataset", "is-valid", {
    "dataset": DATASET,
})

check("Preview dataset", "first-rows", {
    "dataset": DATASET,
    "config": "default",
    "split": "train",
})

check("Uji filter bahasa Indonesia positif", "filter", {
    "dataset": DATASET,
    "config": "default",
    "split": "train",
    "where": '"language"=\'id\' AND "label"=\'positive\'',
    "offset": 0,
    "length": 5,
})

print("\nPemeriksaan selesai.")
