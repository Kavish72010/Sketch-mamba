# download_quickdraw_preprocessed.py
#
# Downloads the TWO preprocessed QuickDraw formats required for StrokeIntent-Lite:
#
#   1. Sketch-RNN .npz  — stroke-3 (Δx, Δy, pen_state), RDP-simplified, pre-split
#      Source : gs://quickdraw_dataset/sketchrnn/{category}.npz
#      Used for: primary sequence modelling (causal Transformer + diffusion head)
#
#   2. Numpy Bitmap .npy — 28×28 greyscale rendered drawings, ~5 MB per category
#      Source : gs://quickdraw_dataset/full/numpy_bitmap/{category}.npy
#      Used for: CNN visual encoder (dual-modal canvas state signal)
#
# Only the 50 semantically diverse categories needed for the project are downloaded
# by default (~6 GB total). Pass --categories to override, or --all for all 345.
#
# Features (preserved from original script):
#   • HTTP Range resume  — resumes interrupted downloads
#   • Exponential backoff — retries up to MAX_RETRIES times
#   • Temp-file staging  — atomic rename on 100% success only
#   • Per-dataset manifest JSON — crash-safe progress tracking
#   • Unified overall progress bar
#
# Required: pip install requests tqdm
#
# Usage:
#   python download_quickdraw_preprocessed.py                  # default 50 cats, both datasets
#   python download_quickdraw_preprocessed.py --dataset npz    # stroke-3 only
#   python download_quickdraw_preprocessed.py --dataset npy    # bitmaps only
#   python download_quickdraw_preprocessed.py --categories cat dog bicycle
#   python download_quickdraw_preprocessed.py --all            # all 345 categories

import os
import json
import time
import logging
import argparse
import requests
from pathlib import Path
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

BASE_OUTPUT_DIR  = Path("./quickdraw_preprocessed")

# Sub-directories per format
NPZ_DIR          = BASE_OUTPUT_DIR / "sketchrnn_npz"    # stroke-3 .npz files
NPY_DIR          = BASE_OUTPUT_DIR / "numpy_bitmap_npy" # 28×28 bitmap .npy files

# Google Cloud Storage HTTP endpoints (no gsutil required)
NPZ_BASE_URL     = "https://storage.googleapis.com/quickdraw_dataset/sketchrnn/{}.npz"
NPY_BASE_URL     = "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/{}.npy"

MAX_RETRIES      = 5
BACKOFF_BASE     = 2        # seconds (doubles each retry)
CHUNK_SIZE       = 1024 * 1024   # 1 MB streaming chunks
TIMEOUT          = (10, 120)     # (connect_timeout, read_timeout)

# ─────────────────────────────────────────────────────────────────────────────
# 50 SEMANTICALLY DIVERSE CATEGORIES FOR STROKEINTENT-LITE
# Covers: animals, vehicles, food, tools/objects, nature, geometry, buildings
# ~3.5M training examples total (50 cats × ~70K drawings each in the .npz files)
# ─────────────────────────────────────────────────────────────────────────────

PROJECT_CATEGORIES = [
    # Animals (10)
    "cat", "dog", "bird", "fish", "horse", "elephant", "bear", "rabbit",
    "butterfly", "spider",
    # Vehicles (6)
    "car", "bicycle", "airplane", "bus", "sailboat", "motorbike",
    # Food (8)
    "apple", "banana", "pizza", "hamburger", "cake", "cookie", "strawberry",
    # Objects & Tools (10)
    "chair", "clock", "knife", "hammer", "key", "umbrella",
    "backpack", "guitar", "telephone", "camera",
    # Nature (8)
    "tree", "flower", "sun", "cloud", "mountain", "snowflake",
    "rainbow", "leaf",
    # Geometry (6)
    "circle", "square", "triangle", "star", "hexagon", "diamond",
    # Buildings & Structures (4) — kept small; high stroke complexity
    "house", "castle", "lighthouse", "bridge",
    # Clothing (4) — useful for ambiguous early-stroke tests
    "hat", "shoe", "t-shirt",
    # Miscellaneous expressive (4)
    "face", "hand", "eye", "crown",
]

# Full 345-category list (used only with --all flag)
ALL_CATEGORIES = [
    "aircraft carrier","airplane","alarm clock","ambulance","angel",
    "animal migration","ant","anvil","apple","arm","asparagus","axe",
    "backpack","banana","bandage","barn","baseball bat","baseball",
    "basket","basketball","bat","bathtub","beach","bear","beard","bed",
    "bee","belt","bench","bicycle","binoculars","bird","birthday cake",
    "blackberry","blueberry","book","boomerang","bottlecap","bowtie",
    "bracelet","brain","bread","bridge","broccoli","broom","bucket",
    "bulldozer","bus","bush","butterfly","cactus","cake","calculator",
    "calendar","camel","camera","camouflage","campfire","candle","cannon",
    "canoe","car","carrot","castle","cat","ceiling fan","cell phone",
    "cello","chair","chandelier","church","circle","clarinet","clock",
    "cloud","coffee cup","compass","computer","cookie","cooler","couch",
    "cow","crab","crayon","crocodile","crown","cruise ship","cup",
    "diamond","dishwasher","diving board","dog","dolphin","donut","door",
    "dragon","dresser","drill","drums","duck","dumbbell","ear","elbow",
    "elephant","envelope","eraser","eye","eyeglasses","face","fan",
    "feather","fence","finger","fire hydrant","fireplace","firetruck",
    "fish","flamingo","flashlight","flip flops","floor lamp","flower",
    "flying saucer","foot","fork","frog","frying pan","garden hose",
    "garden","giraffe","goatee","golf club","grapes","grass","guitar",
    "hamburger","hammer","hand","harp","hat","headphones","hedgehog",
    "helicopter","helmet","hexagon","hockey puck","hockey stick","horse",
    "hospital","hot air balloon","hot dog","hot tub","hourglass",
    "house plant","house","hurricane","ice cream","jacket","jail",
    "kangaroo","key","keyboard","knee","knife","ladder","lantern",
    "laptop","leaf","leg","light bulb","lighter","lighthouse","lightning",
    "line","lion","lipstick","lobster","lollipop","mailbox","map",
    "marker","matches","megaphone","mermaid","microphone","microwave",
    "monkey","moon","mosquito","motorbike","mountain","mouse","moustache",
    "mouth","mug","mushroom","nail","necklace","nose","ocean","octagon",
    "octopus","onion","oven","owl","paint can","paintbrush","palm tree",
    "panda","pants","paper clip","parachute","parrot","passport","peanut",
    "pear","peas","pencil","penguin","piano","pickup truck",
    "picture frame","pig","pillow","pineapple","pizza","pliers",
    "police car","pond","pool","popsicle","postcard","potato",
    "power outlet","purse","rabbit","raccoon","radio","rain","rainbow",
    "rake","remote control","rhinoceros","rifle","river","roller coaster",
    "rollerskates","sailboat","sandwich","saw","saxophone","school bus",
    "scissors","scorpion","screwdriver","sea turtle","see saw","shark",
    "sheep","shoe","shorts","shovel","sink","skateboard","skull",
    "skyscraper","sleeping bag","smiley face","snail","snake","snorkel",
    "snowflake","snowman","soccer ball","sock","speedboat","spider",
    "spoon","spreadsheet","square","squiggle","squirrel","stairs","star",
    "steak","stereo","stethoscope","stitches","stop sign","stove",
    "strawberry","streetlight","string bean","submarine","suitcase","sun",
    "swan","sweater","swing set","sword","syringe","t-shirt","table",
    "teapot","teddy-bear","telephone","television","tennis racquet","tent",
    "The Eiffel Tower","The Great Wall of China","The Mona Lisa","tiger",
    "toaster","toe","toilet","tooth","toothbrush","toothpaste","tornado",
    "tractor","traffic light","train","tree","triangle","trombone",
    "truck","trumpet","umbrella","underwear","van","vase","violin",
    "washing machine","watermelon","waterslide","whale","wheel","windmill",
    "wine bottle","wine glass","wristwatch","yoga","zebra","zigzag",
]

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────

BASE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(BASE_OUTPUT_DIR / "download.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# MANIFEST: per-dataset crash-safe tracking
# ─────────────────────────────────────────────────────────────────────────────

def load_manifest(manifest_path: Path) -> dict:
    if manifest_path.exists():
        with open(manifest_path) as f:
            return json.load(f)
    return {"completed": [], "failed": []}

def save_manifest(manifest: dict, manifest_path: Path):
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

# ─────────────────────────────────────────────────────────────────────────────
# URL HELPERS
# Each format uses a different separator for spaces in category names
# ─────────────────────────────────────────────────────────────────────────────

def npz_url(category: str) -> str:
    """Sketch-RNN .npz: spaces → underscores (Google's naming convention)."""
    return NPZ_BASE_URL.format(category.replace(" ", "_"))

def npy_url(category: str) -> str:
    """Numpy bitmap .npy: spaces → %20 (standard URL encoding)."""
    return NPY_BASE_URL.format(category.replace(" ", "%20"))

# ─────────────────────────────────────────────────────────────────────────────
# CORE DOWNLOAD: resume + retry (identical logic to original script)
# ─────────────────────────────────────────────────────────────────────────────

def download_file(url: str, dest_path: Path, label: str) -> bool:
    """
    Downloads a single file with:
      - HTTP Range resume  → picks up from a partial .part file
      - Exponential backoff → up to MAX_RETRIES attempts
      - Temp-file staging  → atomic rename on success only
    Returns True on success, False if all retries exhausted.
    """
    temp_path = dest_path.with_suffix(dest_path.suffix + ".part")

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            existing_bytes = temp_path.stat().st_size if temp_path.exists() else 0
            headers = {"Range": f"bytes={existing_bytes}-"} if existing_bytes > 0 else {}

            response = requests.get(url, headers=headers, stream=True, timeout=TIMEOUT)

            if response.status_code == 416:
                # Server says the range is already satisfied — file is complete
                temp_path.rename(dest_path)
                return True

            if response.status_code not in (200, 206):
                raise requests.HTTPError(f"HTTP {response.status_code} for {url}")

            total_bytes = int(response.headers.get("Content-Length", 0))
            if response.status_code == 206:
                total_bytes += existing_bytes

            mode = "ab" if existing_bytes > 0 else "wb"
            display_label = f"{label[:32]:<32}"

            with open(temp_path, mode) as f, tqdm(
                desc=display_label,
                total=total_bytes,
                initial=existing_bytes,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                leave=False,
                dynamic_ncols=True,
            ) as bar:
                for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                    if chunk:
                        f.write(chunk)
                        bar.update(len(chunk))

            temp_path.rename(dest_path)
            return True

        except (requests.RequestException, OSError) as e:
            wait = BACKOFF_BASE ** attempt
            log.warning(
                f"[{label}] Attempt {attempt}/{MAX_RETRIES} failed: {e}. "
                f"Retrying in {wait}s..."
            )
            time.sleep(wait)

    log.error(f"[{label}] FAILED after {MAX_RETRIES} attempts. Skipping.")
    return False

# ─────────────────────────────────────────────────────────────────────────────
# DATASET DOWNLOADER
# ─────────────────────────────────────────────────────────────────────────────

DATASET_CONFIG = {
    "npz": {
        "name"        : "Sketch-RNN .npz (stroke-3, pre-split train/val/test)",
        "output_dir"  : NPZ_DIR,
        "manifest"    : BASE_OUTPUT_DIR / "_manifest_npz.json",
        "url_fn"      : npz_url,
        "ext"         : ".npz",
        "approx_size" : "~30–80 MB per category",
        "used_for"    : "Causal Transformer context encoder + MLP diffusion head",
    },
    "npy": {
        "name"        : "Numpy Bitmap .npy (28×28 greyscale)",
        "output_dir"  : NPY_DIR,
        "manifest"    : BASE_OUTPUT_DIR / "_manifest_npy.json",
        "url_fn"      : npy_url,
        "ext"         : ".npy",
        "approx_size" : "~4–8 MB per category",
        "used_for"    : "CNN visual encoder (dual-modal canvas state signal)",
    },
}

def run_dataset(dataset_key: str, categories: list):
    cfg = DATASET_CONFIG[dataset_key]
    out_dir: Path = cfg["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path: Path = cfg["manifest"]
    manifest = load_manifest(manifest_path)
    already_done = set(manifest["completed"])

    pending = [
        c for c in categories
        if c not in already_done and not (out_dir / f"{c}{cfg['ext']}").exists()
    ]

    log.info("─" * 64)
    log.info(f"Dataset  : {cfg['name']}")
    log.info(f"Used for : {cfg['used_for']}")
    log.info(f"Size     : {cfg['approx_size']}")
    log.info(f"Output   : {out_dir.resolve()}")
    log.info(
        f"Categories — Total: {len(categories)} | "
        f"Already done: {len(already_done)} | Pending: {len(pending)}"
    )

    if not pending:
        log.info("All categories already downloaded. Nothing to do.")
        return

    with tqdm(
        total=len(pending),
        desc=f"[{dataset_key.upper()}] Overall",
        unit="file",
        dynamic_ncols=True,
        colour="cyan" if dataset_key == "npz" else "yellow",
    ) as overall_bar:
        for category in pending:
            dest = out_dir / f"{category}{cfg['ext']}"
            url  = cfg["url_fn"](category)
            label = f"[{dataset_key}] {category}"

            success = download_file(url, dest, label)

            if success:
                manifest["completed"].append(category)
                if category in manifest["failed"]:
                    manifest["failed"].remove(category)
                log.info(f"    {category}")
            else:
                if category not in manifest["failed"]:
                    manifest["failed"].append(category)
                log.error(f"    {category}")

            save_manifest(manifest, manifest_path)
            overall_bar.update(1)
            overall_bar.set_postfix(
                done=len(manifest["completed"]),
                failed=len(manifest["failed"]),
            )

    log.info(f"[{dataset_key.upper()}] Done — "
             f" {len(manifest['completed'])}   {len(manifest['failed'])}")
    if manifest["failed"]:
        log.warning(f"  Failed: {manifest['failed']}")
        log.warning("  Re-run to retry only the failed files automatically.")

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Download preprocessed QuickDraw data for StrokeIntent-Lite.\n"
            "Downloads Sketch-RNN .npz (stroke-3) and/or Numpy Bitmap .npy files.\n"
            "Defaults to the 50 project categories with both formats."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python download_quickdraw_preprocessed.py\n"
            "  python download_quickdraw_preprocessed.py --dataset npz\n"
            "  python download_quickdraw_preprocessed.py --dataset npy\n"
            "  python download_quickdraw_preprocessed.py --categories cat dog bicycle\n"
            "  python download_quickdraw_preprocessed.py --all\n"
        ),
    )
    parser.add_argument(
        "--dataset",
        choices=["npz", "npy", "both"],
        default="both",
        help=(
            "Which dataset format to download.\n"
            "  npz  — Sketch-RNN stroke-3 .npz files (primary training data)\n"
            "  npy  — Numpy bitmap 28×28 .npy files (visual encoder input)\n"
            "  both — Download both (default)"
        ),
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=None,
        metavar="CATEGORY",
        help=(
            'Space-separated category names. Defaults to the 50 project categories. '
            'Example: --categories cat dog bicycle "ice cream"'
        ),
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all 345 QuickDraw categories instead of the default 50.",
    )
    args = parser.parse_args()

    # Resolve category list
    if args.categories:
        categories = args.categories
    elif args.all:
        categories = ALL_CATEGORIES
        log.info("--all flag set: downloading all 345 categories.")
    else:
        categories = PROJECT_CATEGORIES
        log.info(f"Using default 50 project categories.")

    log.info(f"Categories selected: {len(categories)}")
    log.info(f"Dataset mode: {args.dataset}")

    datasets_to_run = (
        ["npz", "npy"] if args.dataset == "both"
        else [args.dataset]
    )

    for ds in datasets_to_run:
        run_dataset(ds, categories)

    log.info("=" * 64)
    log.info("All requested downloads complete.")
    log.info(f"Output root: {BASE_OUTPUT_DIR.resolve()}")
    log.info("Next step  : Run your preprocessing pipeline on:")
    log.info(f"  Stroke-3 .npz → {NPZ_DIR.resolve()}")
    log.info(f"  Bitmaps  .npy → {NPY_DIR.resolve()}")


if __name__ == "__main__":
    main()
